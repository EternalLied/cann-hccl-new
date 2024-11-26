/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_gather_half_ring.h"

namespace hccl {
AllGatherHalfRing::AllGatherHalfRing(const HcclDispatcher dispatcher, const u32 commIndex) : ExecutorBase(dispatcher), commIndex_(commIndex)
{
}

AllGatherHalfRing::~AllGatherHalfRing()
{
}

HcclResult AllGatherHalfRing::TxVector(const LINK &link, const std::vector<Slice> &txSlices)
{
    std::vector<TxMemoryInfo> txMems;
    for (const Slice &txSlice : txSlices) {
        DeviceMem srcMem = outputMem_.range(txSlice.offset, txSlice.size);
        HCCL_DEBUG("tx srcMem[%p] range[%llu] size[%llu] ", srcMem.ptr(), txSlice.offset, txSlice.size);
        txMems.emplace_back(TxMemoryInfo{UserMemType::OUTPUT_MEM, txSlice.offset + baseOffset_,
            srcMem.ptr(), txSlice.size});
    }
    CHK_RET(link->TxAsync(txMems, stream_));
    return HCCL_SUCCESS;
}

HcclResult AllGatherHalfRing::RxVector(const LINK &link, const std::vector<Slice> &rxSlices)
{
    std::vector<RxMemoryInfo> rxMems;
    for (const Slice &rxSlice : rxSlices) {
        DeviceMem dstMem = outputMem_.range(rxSlice.offset, rxSlice.size);
        HCCL_DEBUG("rx dstMem[%p] range[%llu], size[%llu] ",  dstMem.ptr(),
            rxSlice.offset, rxSlice.size);
        rxMems.emplace_back(RxMemoryInfo{UserMemType::OUTPUT_MEM, rxSlice.offset + baseOffset_,
            dstMem.ptr(), rxSlice.size});
    }
    CHK_RET(link->RxAsync(rxMems, stream_));
    return HCCL_SUCCESS;
}

HcclResult AllGatherHalfRing::Tx(const LINK &link, const Slice &txSlice)
{
    DeviceMem srcMem = outputMem_.range(txSlice.offset, txSlice.size);
    HCCL_DEBUG("tx srcMem[%p] range[%llu] size[%llu] ", srcMem.ptr(), txSlice.offset, txSlice.size);
    CHK_RET(link->TxAsync(UserMemType::OUTPUT_MEM, txSlice.offset + baseOffset_, srcMem.ptr(), txSlice.size, stream_));
    return HCCL_SUCCESS;
}

HcclResult AllGatherHalfRing::Rx(const LINK &link, const Slice &rxSlice)
{
    DeviceMem dstMem = outputMem_.range(rxSlice.offset, rxSlice.size);
    HCCL_DEBUG("rx dstMem[%p] range[%llu], size[%llu] ",  dstMem.ptr(),
        rxSlice.offset, rxSlice.size);
    CHK_RET(link->RxAsync(UserMemType::OUTPUT_MEM, rxSlice.offset + baseOffset_, dstMem.ptr(), rxSlice.size, stream_));
    return HCCL_SUCCESS;
}

// 服务器间allgather的入口函数
HcclResult AllGatherHalfRing::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links)
{
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    HCCL_INFO("AllGatherHalfRing run_async rank[%u] ranksize[%u] inputMem[%p] outputMem[%p] count[%llu]", \
              rank, rankSize, inputMem_.ptr(), outputMem_.ptr(), count_);

    if (rankSize == 1) {
        if (inputMem_ != outputMem_) {
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, outputMem_, inputMem_, stream_));
        }
        return HCCL_SUCCESS;
    }

    // 获取ring algorithm所需的通信连接
    u32 ringPrevRank = (rank + rankSize - 1) % rankSize;
    u32 ringNextRank = (rank + 1) % rankSize;

    if (links.size() < rankSize) {
        HCCL_ERROR("[AllGatherHalfRing][RunAsync]rank[%u] linkSize is less than rankSize", rank);
        return HCCL_E_INTERNAL;
    }

    linkLeft_ = links[ringPrevRank];
    CHK_SMART_PTR_NULL(linkLeft_);

    linkRight_ = links[ringNextRank];
    CHK_SMART_PTR_NULL(linkRight_);

    u32 unitSize = DataUnitSize(dataType_);
    if (unitSize == 0) {
        HCCL_ERROR("[AllGatherHalfRing][RunAsync]unitSize is zero");
        return HCCL_E_INTERNAL;
    }

    std::vector<Slice> inputSlices(slices_);
    if (slices_.size() == 0) {
        slices_.resize(rankSize);
        inputSlices.resize(rankSize);

        u64 sliceSize = count_ * unitSize;
        for (u32 i = 0; i < rankSize; i++) {
            slices_[i].size = sliceSize;
            slices_[i].offset = sliceSize * i;
            inputSlices[i].size = sliceSize;
            inputSlices[i].offset = (inputMem_.size() < outputMem_.size()) ? 0 : (sliceSize * i);
            HCCL_DEBUG("rank[%u], slices[%u].offset=%llu, slices[%u].size=%llu", \
                       rank, i, slices_[i].offset, i, slices_[i].size);
        }
    }

    // 双buffer下, 先将input拷贝到output的合适位置
    if (inputMem_ != outputMem_) {
        DeviceMem dst = outputMem_.range(slices_[rank].offset, slices_[rank].size);
        DeviceMem src = inputMem_.range(inputSlices[rank].offset, inputSlices[rank].size);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));
    }

    // 运行all-gather, ring算法
    // 单环场景下 nicRankList_ 长度默认为 8。
    // 多环场景下 nicRankList_ 长度为网口数量。此时若 rankSize != nicRankList_ 则为网口裁剪场景
    if (rankSize != HCCL_NIC_MAX_NUM || nicRankList_.size() == HCCL_NIC_MAX_NUM) {
        // 非网口裁剪场景:
        CHK_RET(RunAllGather(rank, rankSize, slices_));
    } else {
        // 网口裁剪场景：当前仅在 910A 8P_RING (4环)，且网口不满配情况下使用
        CHK_RET(AllGatherSlicesPrep(rankSize, nicRankList_.size()));
        CHK_RET(RunAllGatherChunk(rank, rankSize, slices_));
    }

    if (barrierSwitchOn_) {
        // 执行barrier，保证数据收发完成
        CHK_RET(ExecuteBarrier(linkLeft_, linkRight_));
    }

    HCCL_INFO("AllGatherHalfRing finished: rank[%u] end", rank);
    return HCCL_SUCCESS;
}

HcclResult AllGatherHalfRing::RunAllGather(u32 rank, u32 rankSize, const std::vector<Slice> &outputSlices, const std::vector<Slice> &outputSlices)
{
    if (outputSlices.size() < rankSize) {
        HCCL_ERROR("[Run][AllGather]rank[%u] OutputSlice Size is less than rank size", rank);
        return HCCL_E_INTERNAL;
    }
    HcclResult ret = HCCL_SUCCESS;

    // 首次传输，将本rank的数据发送到下游
    u32 sliceSize = outputSlices.size() / rankSize;
    u32 rxSliceIndex = ForwordRank(rank, rankSize, 1);
    u32 txSliceIndex = rank;
    for (u32 i = 0; i < rankSize / 2; i++) {
        HCCL_DEBUG("rank[%u] round[%u] will tx_ack  outputslice[%u].offset is[%llu] size[%llu]",
            rank, i, rxSliceIndex, outputSlices[rxSliceIndex].offset, outputSlices[rxSliceIndex].size);
        CHK_RET(linkLeft_->TxAck(stream_));

        // reduce目的操作
        HCCL_DEBUG("rank[%u] round[%u] will rx ack because outputSlices[%u] size[%llu] ", rank, \
            i, txSliceIndex, outputSlices[txSliceIndex].size);
        CHK_RET(linkRight_->RxAck(stream_));

        std::vector<Slice> txSegsSlice;
        std::vector<Slice> rxSegsSlice;
        if (i < rankSize / 2 - 1) {
            for (u32 j = 0; j < sliceSize; j++) {
                txSegsSlice.push_back(outputSlices[txSliceIndex * sliceSize + j]);
                rxSegsSlice.push_back(outputSlices[rxSliceIndex * sliceSize + j]);
            }
        } else {
            Slice txTempSlice, rxTempSlice;
            for (u32 j = 0; j < sliceSize; j++) {
                    u32 txTempIndex = txSliceIndex * sliceSize + j;
                    u32 rxTempIndex = rxSliceIndex * sliceSize + j;
                    if (commIndex_ == 0) {
                        txTempSlice.offset = outputSlices[txTempIndex].offset + outputSlices[txTempIndex].size / 2;
                        rxTempSlice.offset = outputSlices[rxTempIndex].offset + outputSlices[rxTempIndex].size / 2;
                    } else {
                        txTempSlice.offset = outputSlices[txTempIndex].offset;
                        rxTempSlice.offset = outputSlices[rxTempIndex].offset;
                    }
                    txTempSlice.size = outputSlices[txTempIndex].size / 2;
                    rxTempSlice.size = outputSlices[rxTempIndex].size / 2;
                    txSegsSlice.push_back(txTempSlice);
                    rxSegsSlice.push_back(rxTempSlice);
            }
        }

        ret = TxVector(linkRight_, txSegsSlice);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Run][AllGather]rank[%u] round[%u] Right Link tx outputSlices[%u] "\
                "Failed", rank, i, txSliceIndex), ret);

        // reduce源操作
        HCCL_DEBUG("rank[%u]  round[%u] rx data outputSlices[%u] offset[%llu] size[%llu]", \
            rank, i, rxSliceIndex, outputSlices[rxSliceIndex].offset, outputSlices[rxSliceIndex].size);
        ret = RxVector(linkLeft_, rxSegsSlice);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Run][AllGather]rank[%u] round[%u]  Left Link rx outputSlices[%u] "\
                "Failed", rank, i, rxSliceIndex), ret);

        // 末尾传输, 只接收一次, 不用再次发送
        txSliceIndex = ForwordRank(txSliceIndex, rankSize, 1);
        rxSliceIndex = ForwordRank(rxSliceIndex, rankSize, 1);

        ret = linkLeft_->RxWaitDone(stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][ReduceScatter]RxWaitDone failed"), ret);
        ret = linkRight_->TxWaitDone(stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][ReduceScatter]TxWaitDone failed"), ret);
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherHalfRing::RunAllGatherChunk(const u32 rank, const u32 rankSize,
                                            const std::vector<Slice> &outputSlices)
{
    if (outputSlices.size() < rankSize) {
        HCCL_ERROR("[Run][AllGatherChunk]rank[%u] OutputSlice Size is less than rank size", rank);
        return HCCL_E_INTERNAL;
    }
    HcclResult ret;
    u32 sendSliceLen = rankSliceLists_[rank].size();
    u32 chunkSize = HCCL_NIC_MAX_NUM / nicRankList_.size();
    if (sendSliceLen >= chunkSize) {
        CHK_RET(HeadAllGatherChunk(rank, rankSize, outputSlices));
        for (u32 midRankIdx = 1; midRankIdx < sendSliceLen - 1; midRankIdx++) {
            ret = MidAllGatherChunk(rank, rankSize, midRankIdx, outputSlices);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Run][AllGatherChunk]rank[%u] run mid[%u] reduce scatter chunk failed", rank, midRankIdx),
                HCCL_E_INTERNAL);
        }
        CHK_RET(TailAllGatherChunk(rank, rankSize, sendSliceLen - 1, outputSlices));
    } else {
        for (u32 rxSliceIndex = 0; rxSliceIndex < HCCL_NIC_MAX_NUM; rxSliceIndex++) {
            CHK_RET(linkLeft_->TxAck(stream_));

            ret = Rx(linkLeft_, outputSlices[rxSliceIndex]);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][AllGatherChunk]rank[%u] Left Link rx outputSlices[%u] "\
                "Failed", rank, rxSliceIndex), ret);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherHalfRing::HeadAllGatherChunk(u32 rank, u32 rankSize, const std::vector<Slice> &outputSlices)
{
    if (outputSlices.size() < rankSize) {
        HCCL_ERROR("[AllGatherHalfRing][HeadAllGatherChunk]rank[%u] OutputSlice Size is less than rank size", rank);
        return HCCL_E_INTERNAL;
    }
    HcclResult ret;
    u32 rxSliceIndex = rankSliceLists_[rank][0];
    u32 txSliceIndex = rxSliceIndex;
    std::vector<u32> preRankSlices(rankSliceLists_[(rank - 1 + rankSize) % rankSize]);
    std::vector<u32>::iterator iterSlice = std::find(preRankSlices.begin(), preRankSlices.end(), rxSliceIndex);
    if (iterSlice != preRankSlices.end()) {
        CHK_RET(linkLeft_->TxAck(stream_));

        ret = Rx(linkLeft_, outputSlices[rxSliceIndex]);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AllGatherHalfRing][HeadAllGatherChunk]rank[%u] Left Link rx "\
            "outputSlices[%u] Failed", rank, rxSliceIndex), ret);
    }

    iterSlice = std::find(preRankSlices.begin(), preRankSlices.end(), rankSliceLists_[rank][1]);
    if (iterSlice != preRankSlices.end()) {
        CHK_RET(MidAllGatherChunk(rank, rankSize, 0, outputSlices));
    } else {
        CHK_RET(linkRight_->RxAck(stream_));

        ret = Tx(linkRight_, outputSlices[txSliceIndex]);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AllGatherHalfRing][HeadAllGatherChunk]rank[%u] Right Link tx "\
            "outputSlices[%u] Failed", rank, txSliceIndex), ret);
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherHalfRing::MidAllGatherChunk(u32 rank, u32 rankSize, u32 sliceIdx,
                                            const std::vector<Slice> &outputSlices)
{
    if (outputSlices.size() < rankSize) {
        HCCL_ERROR("[AllGatherHalfRing][MidAllGatherChunk]rank[%u] OutputSlice Size is less than rank size", rank);
        return HCCL_E_INTERNAL;
    }
    HcclResult ret;
    u32 rxSliceIndex = rankSliceLists_[rank][sliceIdx + 1];
    u32 txSliceIndex = rankSliceLists_[rank][sliceIdx];
    std::vector<u32> preRankSlices(rankSliceLists_[(rank - 1 + rankSize) % rankSize]);
    std::vector<u32>::iterator iterSlice = std::find(preRankSlices.begin(), preRankSlices.end(), rxSliceIndex);
    if (iterSlice != preRankSlices.end()) {
        CHK_RET(linkLeft_->TxAck(stream_));

        CHK_RET(linkRight_->RxAck(stream_));

        ret = Tx(linkRight_, outputSlices[txSliceIndex]);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AllGatherHalfRing][MidAllGatherChunk]rank[%u] Right Link tx "\
            "outputSlices[%u] Failed", rank, txSliceIndex), ret);
        ret = Rx(linkLeft_, outputSlices[rxSliceIndex]);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AllGatherHalfRing][MidAllGatherChunk]rank[%u] Left Link rx "\
            "outputSlices[%u] Failed", rank, rxSliceIndex), ret);
    } else {
        CHK_RET(linkRight_->RxAck(stream_));

        ret = Tx(linkRight_, outputSlices[txSliceIndex]);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AllGatherHalfRing][MidAllGatherChunk]rank[%u] Right Link tx "\
            "outputSlices[%u] Failed", rank, txSliceIndex), ret);
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherHalfRing::TailAllGatherChunk(u32 rank, u32 rankSize, u32 sliceIdx,
                                             const std::vector<Slice> &outputSlices)
{
    if (outputSlices.size() < rankSize) {
        HCCL_ERROR("[AllGatherHalfRing][TailAllGatherChunk]rank[%u] OutputSlice Size is less than rank size", rank);
        return HCCL_E_INTERNAL;
    }
    HcclResult ret;
    u32 chunkSize = HCCL_NIC_MAX_NUM / nicRankList_.size();
    u32 txSliceIndex = rankSliceLists_[rank][sliceIdx];
    u32 nextRank = (rank + 1 + rankSize) % rankSize;
    std::vector<u32>::iterator iterNic = std::find(nicRankList_.begin(), nicRankList_.end(), nextRank);
    if (iterNic != nicRankList_.end()) {
        u32 nicIdx = distance(nicRankList_.begin(), iterNic);
        u32 chunkStart = nicIdx * chunkSize;
        u32 rxSliceIndex = chunkStart;
        CHK_RET(linkLeft_->TxAck(stream_));

        CHK_RET(linkRight_->RxAck(stream_));

        ret = Tx(linkRight_, outputSlices[txSliceIndex]);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AllGatherHalfRing][TailAllGatherChunk]rank[%u] Right Link tx "\
            "outputSlices[%u] Failed", rank, txSliceIndex), ret);
        ret = Rx(linkLeft_, outputSlices[rxSliceIndex]);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AllGatherHalfRing][TailAllGatherChunk]rank[%u] Left Link rx "\
            "outputSlices[%u] Failed", rank, rxSliceIndex), ret);

        for (u32 sliceIdx = 1; sliceIdx < chunkSize; sliceIdx++) {
            rxSliceIndex = chunkStart + sliceIdx;
            CHK_RET(linkLeft_->TxAck(stream_));

            ret = Rx(linkLeft_, outputSlices[rxSliceIndex]);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AllGatherHalfRing][TailAllGatherChunk]rank[%u] Left Link rx "\
                "outputSlices[%u] Failed", rank, rxSliceIndex), ret);
        }
    } else {
        CHK_RET(linkRight_->RxAck(stream_));

        ret = Tx(linkRight_, outputSlices[txSliceIndex]);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AllGatherHalfRing][TailAllGatherChunk]rank[%u] Right Link tx "\
            "outputSlices[%u] Failed", rank, txSliceIndex), ret);
    }
    return HCCL_SUCCESS;
}

// 多网口allgather各rank发送slice准备
HcclResult AllGatherHalfRing::AllGatherSlicesPrep(u32 rankSize, u32 nicSize)
{
    u32 chunkSize = HCCL_NIC_MAX_NUM / nicSize;
    for (u32 rankIdx = 0; rankIdx < rankSize; rankIdx++) {
        std::vector<u32> sliceList;    // 单个rank上的发送slice编号
        for (u32 nicDis = 0; nicDis <= rankSize - 2; nicDis++) {  // 递减从当前rank遍历至(rank+2+ranksize)%ranksize的位置
            u32 nicIdx = (rankIdx + rankSize - nicDis) % rankSize;
            std::vector<u32>::iterator iterNic = std::find(nicRankList_.begin(), nicRankList_.end(), nicIdx);
            if (iterNic != nicRankList_.end()) {    // 当前rank为网口所在位置，将网口对应的chunksize份silce放入sliceList
                u32 nicListIdx = distance(nicRankList_.begin(), iterNic);
                for (u32 chunkIdx = 0; chunkIdx < chunkSize; chunkIdx++) {
                    sliceList.push_back(chunkSize * nicListIdx + chunkIdx);
                }
            }
        }
        rankSliceLists_.push_back(sliceList);
    }
    return HCCL_SUCCESS;
}
}  // namespace hccl
