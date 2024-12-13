/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "all_gather_half_ring_direct.h"

namespace hccl {
AllGatherHalfRingDirect::AllGatherHalfRingDirect(
    const HcclDispatcher dispatcher, const HcomCollOpInfo *opInfo, const u32 userRank,
    std::vector<Stream> &subStreams, const std::vector<std::shared_ptr<LocalNotify>> &mainSignals,
    const std::vector<std::shared_ptr<LocalNotify>> &subSignals, const std::vector<u32> &ringsOrder,
    const u32 commIndex, const std::vector<Slice> &userMemOutputSlices, bool isSdma)
    : ExecutorBase(dispatcher), opInfo_(opInfo), userRank_(userRank), subStreams_(subStreams),
      mainSignals_(mainSignals), subSignals_(subSignals), ringsOrder_(ringsOrder), commIndex_(commIndex),
      userMemOutputSlices_(userMemOutputSlices), isSdma_(isSdma)
{
}

AllGatherHalfRingDirect::~AllGatherHalfRingDirect()
{
}

// 服务器间allgather的入口函数
HcclResult AllGatherHalfRingDirect::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links)
{
    // 基本的检查
    CHK_RET(CheckParameters(rank, rankSize, links));

    if (rankSize == 1) {
        CHK_RET(OneRankMemcpy());
        return HCCL_SUCCESS;
    }
    // 收集邻居信息
    CHK_RET(GetInitializedNeighborLinks(rank, rankSize, links));

    // 填充slice_
    CHK_RET(SetSlices(rank, rankSize));

    // 运行all-gather, ring算法
    CHK_RET(RunAllGather(rank, rankSize));

    if (barrierSwitchOn_) {
        // 执行barrier，保证数据收发完成
        CHK_RET(ExecuteBarrier(leftLink_, rightLink_));
    }

    HCCL_INFO("AllGatherHalfRingDirect finished: rank[%u] end", rank);
    return HCCL_SUCCESS;
}

HcclResult AllGatherHalfRingDirect::CheckParameters(const u32 rank, const u32 rankSize,
                                                          const std::vector<LINK> &links)
{
    CHK_PTR_NULL(opInfo_);
    CHK_RET(CheckConcurrentDirectParameters(rank, rankSize, links));
    // 判断subStreams数量是否正确
    CHK_PRT_RET(subStreams_.size() < 1,
                HCCL_ERROR("[AllGatherHalfRingDirect] subStreams size[%u] is less than 1", subStreams_.size()),
                HCCL_E_PARA);
    for (auto &s : subStreams_) {
        CHK_PTR_NULL(s.ptr());
    }
    // 判断mainSignals数量是否正确
    CHK_PRT_RET(mainSignals_.size() < 1,
                HCCL_ERROR("[AllGatherHalfRingDirect] mainSignals size[%u] is less than 1", mainSignals_.size()),
                HCCL_E_PARA);
    // 判断subSignals数量是否正确
    CHK_PRT_RET(subSignals_.size() < 1,
                HCCL_ERROR("[AllGatherHalfRingDirect] subSignals size[%u] is less than 1", subSignals_.size()),
                HCCL_E_PARA);
    // 判断ringsOrder数量是否正确
    CHK_PRT_RET(ringsOrder_.size() % rankSize != 0,
                HCCL_ERROR("[AllGatherHalfRingDirect] ringsOrder size[%u] can not be divided by rank size[%u]",
                    ringsOrder_.size(), rankSize), HCCL_E_PARA);
    // 判断userMemInputSlices数量是否正确
    CHK_PRT_RET(userMemOutputSlices_.size() % rankSize != 0,
        HCCL_ERROR("[AllGatherHalfRingDirect] userMemOutputSlices size[%u] can not be divided by rank size[%u]",
            userMemOutputSlices_.size(), rankSize), HCCL_E_PARA);
    HCCL_INFO("AllGatherHalfRingDirect finished to CheckParameters");
    return HCCL_SUCCESS;
}

HcclResult AllGatherHalfRingDirect::OneRankMemcpy()
{
    for (u32 sliceIdx = 0; sliceIdx < slices_.size(); sliceIdx++) {
        const Slice &srcSlice = slices_[sliceIdx];
        const Slice &dstSlice = userMemOutputSlices_[sliceIdx];
        DeviceMem    src;
        DeviceMem    dst = DeviceMem::create(static_cast<u8 *>(opInfo_->outputAddr) + dstSlice.offset, dstSlice.size);
        if (opInfo_->inputAddr != nullptr) {
            // opInfo_->inputAddr != nullptr指示要从user input获取输入
            u64 stepOffset = slices_[ringsOrder_[0]].offset;
            HCCL_DEBUG("Memcpy operation: stream[main], rank[%u] starts to copy offset[%llu], size[%llu] at userInput",
                userRank_, stepOffset, srcSlice.size);
            src = DeviceMem::create(static_cast<u8 *>(opInfo_->inputAddr) + stepOffset, srcSlice.size);
        } else {
            // opInfo_->inputAddr == nullptr指示要从CCL buffer获取输入
            HCCL_DEBUG("Memcpy operation: stream[main], rank[%u] starts to copy offset[%llu], size[%llu] at inputMem_",
                userRank_, srcSlice.offset, srcSlice.size);
            src = inputMem_.range(srcSlice.offset, srcSlice.size);
        }
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));
    }

    return HCCL_SUCCESS;
}

HcclResult AllGatherHalfRingDirect::GetInitializedNeighborLinks(const u32 rank, const u32 rankSize,
                                                                      const std::vector<LINK> &links)
{
    // 收集左邻居信息
    leftLink_ = links[(rank + rankSize - 1) % rankSize];
    CHK_SMART_PTR_NULL(leftLink_);

    // 收集右邻居信息
    rightLink_ = links[(rank + 1) % rankSize];
    CHK_SMART_PTR_NULL(rightLink_);
    HCCL_INFO("AllGatherHalfRingDirect finished to GetInitializedNeighborLinks");
    return HCCL_SUCCESS;
}

HcclResult AllGatherHalfRingDirect::SetSlices(const u32 rank, const u32 rankSize)
{
    inputSlices_ = slices_;
    if (slices_.size() == 0) {
        slices_.resize(rankSize);
        inputSlices_.resize(rankSize);

        u64 sliceSize = count_ * DataUnitSize(dataType_);
        for (u32 i = 0; i < rankSize; i++) {
            slices_[i].size        = sliceSize;
            slices_[i].offset      = sliceSize * i;
            inputSlices_[i].size   = sliceSize;
            inputSlices_[i].offset = (inputMem_.size() < outputMem_.size()) ? 0 : (sliceSize * i);
            HCCL_DEBUG("rank[%u], slices[%u].offset=%llu, slices[%u].size=[%llu]", rank, i, slices_[i].offset, i,
                       slices_[i].size);
        }
    }
    if (UNLIKELY(HcclCheckLogLevel(HCCL_LOG_DEBUG) == 1)) {
        for (u32 i = 0; i < slices_.size(); i++) {
            HCCL_DEBUG(
                "[AllGatherHalfRingDirect][SetSlices]rank[%u], slices[%u].offset=[%llu], slices[%u].size=[%llu]",
                rank, i, slices_[i].offset, i, slices_[i].size);
        }
    }
    HCCL_INFO("AllGatherHalfRingDirect finished to SetSlices");
    return HCCL_SUCCESS;
}

HcclResult AllGatherHalfRingDirect::RunInitStep(const u32 rank, const u32 rankSize)
{
    // 第一步搬到userMemIn_的offset, 不同的ring环offset不一样
    auto firstStepOffset = slices_[ringsOrder_[0]].offset;
    // 第-1步，片内将部分数据从userIn搬到cclIn
    DeviceMem srcInit;
    DeviceMem dstInit;
    u32 initSliceIdx = rank;
    u32 sliceSize = slices_.size() / rankSize;
    for (u32 sliceIdx = 0; sliceIdx < sliceSize; sliceIdx++) {
        Slice initSlice = slices_[initSliceIdx * sliceSize + sliceIdx];
        // 需要+userMemIn_的offset
        if (opInfo_->inputAddr != nullptr) {
            // AllGather算子调用AllGatherHalfRingDirect场景
            srcInit = DeviceMem::create(static_cast<u8 *>(opInfo_->inputAddr) + firstStepOffset, initSlice.size);
        } else {
            // AllReduce算子调用AllGatherHalfRingDirect场景
            srcInit = inputMem_.range(initSlice.offset, initSlice.size);
        }
        dstInit = outputMem_.range(initSlice.offset, initSlice.size);
        HCCL_DEBUG("Memcpy operation: step[-1] stream[main] src rank[%u] starts to copy(rcv) offset[%llu], "
            "size[%llu] on userMemOutput to offset[%llu], size[%llu] on CCL",
            userRank_, firstStepOffset, initSlice.size, initSlice.offset, initSlice.size);
        // 若src与dst一样，则不需要搬运
        if (srcInit != dstInit) {
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstInit, srcInit, stream_));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherHalfRingDirect::RunAllGather(const u32 rank, const u32 rankSize)
{
    HCCL_INFO("AllGatherHalfRingDirect starts, the input param rank[%u]", rank);
    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));

    CHK_RET(RunInitStep(rank, rankSize));
    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));
    CHK_RET(MainRecordSub()); // 主流通知从流开始通信
    CHK_RET(SubWaitMain());   // 从流等待主流通知

    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));
    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem_, outputMem_, subStreams_[0], dispatcher_));
    u32 txSliceIdx = rank;
    u32 sliceSize = slices_.size() / rankSize;
    u32 rxSliceIdx = (rank + rankSize - 1) % rankSize;

    std::vector<DeviceMem> finalSrc;
    std::vector<DeviceMem> finalDst;
    for (u32 step = 0; step < rankSize / 2; step++) {
        std::vector<Slice> rxSliceVector;
        std::vector<Slice> mainSliceVector;
        std::vector<Slice> txSliceVector;
        std::vector<Slice> subSliceVector;
        if (step < rankSize / 2 - 1){
            for (u32 sliceIdx = 0; sliceIdx < sliceSize; sliceIdx++) {
                rxSliceVector.push_back(slices_[rxSliceIdx * sliceSize + sliceIdx]);
                mainSliceVector.push_back(userMemOutputSlices_[rxSliceIdx * sliceSize + sliceIdx]);
                txSliceVector.push_back(slices_[txSliceIdx * sliceSize + sliceIdx]);
                subSliceVector.push_back(userMemOutputSlices_[txSliceIdx * sliceSize + sliceIdx]);
            }
        } else {
            Slice rxTempSlice, mainTempSlice, txTempSlice, subTempSlice;
            for (u32 sliceIdx = 0; sliceIdx < sliceSize; sliceIdx++) {
                    u32 rxTempIndex = rxSliceIdx * sliceSize + sliceIdx;
                    u32 txTempIndex = txSliceIdx * sliceSize + sliceIdx;
                    if (commIndex_ == 0) {
                        rxTempSlice.offset = slices_[rxTempIndex].offset + slices_[rxTempIndex].size / 2;
                        mainTempSlice.offset = userMemOutputSlices_[rxTempIndex].offset + userMemOutputSlices_[rxTempIndex].size / 2;
                        txTempSlice.offset = slices_[txTempIndex].offset + slices_[txTempIndex].size / 2;
                        subTempSlice.offset = userMemOutputSlices_[txTempIndex].offset + userMemOutputSlices_[txTempIndex].size / 2;
                    } else {
                        rxTempSlice.offset = slices_[rxTempIndex].offset;
                        mainTempSlice.offset = userMemOutputSlices_[rxTempIndex].offset;
                        txTempSlice.offset = slices_[txTempIndex].offset;
                        subTempSlice.offset = userMemOutputSlices_[txTempIndex].offset;
                    }
                    rxTempSlice.size = slices_[rxTempIndex].size / 2;
                    mainTempSlice.size = userMemOutputSlices_[rxTempIndex].size / 2;
                    txTempSlice.size = slices_[txTempIndex].size / 2;
                    subTempSlice.size = userMemOutputSlices_[txTempIndex].size / 2;
                    rxSliceVector.push_back(rxTempSlice);
                    mainSliceVector.push_back(mainTempSlice);
                    txSliceVector.push_back(txTempSlice);
                    subSliceVector.push_back(subTempSlice);
            }
        }
        // 从流
        if (!isSdma_) {
            CHK_RET(LocalNotify::Post(subStreams_[0], dispatcher_, mainSignals_[0], profilerInput_.stage));
            CHK_RET(LocalNotify::Wait(subStreams_[0], dispatcher_, subSignals_[0], profilerInput_.stage));
        }
        for (u32 sliceIdx = 0; sliceIdx < sliceSize; sliceIdx++) {
            if (isSdma_) {
                CHK_RET(LocalNotify::Post(subStreams_[0], dispatcher_, mainSignals_[0], profilerInput_.stage));
                CHK_RET(LocalNotify::Wait(subStreams_[0], dispatcher_, subSignals_[0], profilerInput_.stage));
            }
            DeviceMem src = outputMem_.range(txSliceVector[sliceIdx].offset, txSliceVector[sliceIdx].size);
            DeviceMem dst = DeviceMem::create(static_cast<u8 *>(opInfo_->outputAddr) + subSliceVector[sliceIdx].offset,
                subSliceVector[sliceIdx].size);
            HCCL_DEBUG("Memcpy operation: step[%u] stream[sub], src rank[%u] starts to send offset[%llu] size[%llu], "
                "dst rank starts to rcv offset[%llu] size[%llu] at userMemOutput_",
                step, userRank_, subSliceVector[sliceIdx].offset, subSliceVector[sliceIdx].size,
                txSliceVector[sliceIdx].offset, txSliceVector[sliceIdx].size);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, subStreams_[0]));
        }

        // 主流
        // Ack
        CHK_RET(leftLink_->TxAck(stream_));
        CHK_RET(rightLink_->RxAck(stream_));

        std::vector<TxMemoryInfo> txMems;
        std::vector<RxMemoryInfo> rxMems;
        for (u32 sliceIdx = 0; sliceIdx < sliceSize; sliceIdx++) {
            DeviceMem src = outputMem_.range(txSliceVector[sliceIdx].offset, txSliceVector[sliceIdx].size);
            HCCL_DEBUG("tx srcMem[%p] range[%llu] size[%llu] ", src.ptr(),
                txSliceVector[sliceIdx].offset, txSliceVector[sliceIdx].size);
            txMems.emplace_back(TxMemoryInfo{UserMemType::OUTPUT_MEM, txSliceVector[sliceIdx].offset + baseOffset_,
                src.ptr(), txSliceVector[sliceIdx].size});
            DeviceMem dst;
            if (isSdma_ && step == rankSize - DMA_REDUCE_TWO_OFFSET) {
                HCCL_DEBUG(
                "DMAReduce(sdma) MemcpyAsync operation: step[%u] stream[main], dst rank[%u] starts to rcv "
                "offset[%llu] size[%llu] at userMemOutput_",
                step, userRank_, mainSliceVector[sliceIdx].offset, mainSliceVector[sliceIdx].size);
                dst = DeviceMem::create(static_cast<u8 *>(opInfo_->outputAddr) + mainSliceVector[sliceIdx].offset,
                    mainSliceVector[sliceIdx].size);
            } else {
                HCCL_DEBUG(
                    "MemcpyAsync operation: step[%u] stream[main], dst rank[%u] starts to rcv offset[%llu] size[%llu] "
                    "at outputMem_",
                    step, userRank_, rxSliceVector[sliceIdx].offset, rxSliceVector[sliceIdx].size);
                dst = outputMem_.range(rxSliceVector[sliceIdx].offset, rxSliceVector[sliceIdx].size);
                if (!isSdma_ && step == rankSize - DMA_REDUCE_TWO_OFFSET) {
                    HCCL_DEBUG("DMAReduce(rdma) record final addr");
                    finalSrc.push_back(outputMem_.range(rxSliceVector[sliceIdx].offset, rxSliceVector[sliceIdx].size));
                    finalDst.push_back(DeviceMem::create(static_cast<u8 *>(opInfo_->outputAddr) + 
                    mainSliceVector[sliceIdx].offset, mainSliceVector[sliceIdx].size));
                }
            }
            rxMems.emplace_back(RxMemoryInfo{UserMemType::OUTPUT_MEM, rxSliceVector[sliceIdx].offset + baseOffset_,
                dst.ptr(), rxSliceVector[sliceIdx].size});
        }
        CHK_RET(rightLink_->TxAsync(txMems, stream_));
        if (!isSdma_) {
            CHK_RET(LocalNotify::Wait(stream_, dispatcher_, mainSignals_[0], profilerInput_.stage));
            CHK_RET(ExecutorBase::ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));
            CHK_RET(LocalNotify::Post(stream_, dispatcher_, subSignals_[0], profilerInput_.stage));
            CHK_RET(leftLink_->RxAsync(rxMems, stream_));
        } else {
            CHK_RET(leftLink_->RxDataSignal(stream_));
            for (auto& mem : rxMems) {
                CHK_RET(LocalNotify::Wait(stream_, dispatcher_, mainSignals_[0], profilerInput_.stage));
                CHK_RET(ExecutorBase::ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));
                CHK_RET(LocalNotify::Post(stream_, dispatcher_, subSignals_[0], profilerInput_.stage));
                CHK_PTR_NULL(mem.dst);
                void *srcMemPtr = nullptr;
                CHK_RET(leftLink_->GetRemoteMem(mem.srcMemType, &srcMemPtr));

                DeviceMem srcDevMem(static_cast<s8 *>(srcMemPtr) + mem.srcOffset, mem.len);
                DeviceMem dstDevMem(static_cast<s8 *>(mem.dst), mem.len);
                CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstDevMem, srcDevMem,
                    stream_, leftLink_->GetRemoteRank(), leftLink_->GetLinkType()));
            }
        }

        // 更新索引
        txSliceIdx = (txSliceIdx + rankSize - 1) % rankSize;
        rxSliceIdx = (rxSliceIdx + rankSize - 1) % rankSize;
    }
    CHK_RET(SubRecordMain()); // 从流通知主流通信完成
    CHK_RET(MainWaitSub());   // 主流等待从流通知
    if (!isSdma_) {
        for (u32 vecIdx = 0; vecIdx < finalSrc.size(); vecIdx++) {
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, finalDst[vecIdx], finalSrc[vecIdx], stream_));
        }
    }
    HCCL_INFO("AllGatherHalfRingDirect finished to RunAllGather");
    return HCCL_SUCCESS;
}

// 主流通知从流干活
HcclResult AllGatherHalfRingDirect::MainRecordSub()
{
    for (u32 signalIndex = 0; signalIndex < subSignals_.size(); signalIndex++) {
        CHK_RET(LocalNotify::Post(stream_, dispatcher_, subSignals_[signalIndex],
            profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}
// 从流等待主流
HcclResult AllGatherHalfRingDirect::SubWaitMain()
{
    for (u32 streamIndex = 0; streamIndex < subSignals_.size(); streamIndex++) {
        CHK_RET(LocalNotify::Wait(subStreams_[streamIndex], dispatcher_, subSignals_[streamIndex],
            profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}
// 主流等待从流
HcclResult AllGatherHalfRingDirect::MainWaitSub()
{
    for (u32 signalIndex = 0; signalIndex < mainSignals_.size(); signalIndex++) {
        CHK_RET(LocalNotify::Wait(stream_, dispatcher_, mainSignals_[signalIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}
// 从流告诉主流活干完了
HcclResult AllGatherHalfRingDirect::SubRecordMain()
{
    for (u32 streamIndex = 0; streamIndex < mainSignals_.size(); streamIndex++) {
        CHK_RET(LocalNotify::Post(subStreams_[streamIndex], dispatcher_, mainSignals_[streamIndex],
            profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}
} // namespace hccl
