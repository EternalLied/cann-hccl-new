/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alltoallv_pairwise_pro.h"
#include "externalinput_pub.h"

// 改为AlltoAllVPairWisePro

namespace hccl {
AlltoAllVPairWisePro::AlltoAllVPairWisePro(const HcclDispatcher dispatcher,
    const std::map<u32, std::vector<u64>> &rankSendDisplsMap,
    const std::map<u32, std::vector<u64>> &rankRecvDisplsMap,
    HcclWorkflowMode workMode)
    : dispatcher_(dispatcher), scratchMemSize_(0), sendDataUnitBytes_(0), recvDataUnitBytes_(0),
    rankSendDisplsMap_(rankSendDisplsMap), rankRecvDisplsMap_(rankRecvDisplsMap), workMode_(workMode),
    isAlltoAllZCopyMode_(false)
{}

AlltoAllVPairWisePro::~AlltoAllVPairWisePro() {}

HcclResult AlltoAllVPairWisePro::Prepare(AlltoAllVBufferInfo& sendBuffer, AlltoAllVBufferInfo& recvBuffer,
    bool isAlltoAllZCopyMode, const Stream &stream)
{
    DeviceMem scratchInputMem = DeviceMem();
    DeviceMem scratchOutputMem = DeviceMem();
    CHK_RET(Prepare(sendBuffer, recvBuffer, scratchInputMem, scratchOutputMem, isAlltoAllZCopyMode, stream));
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVPairWisePro::Prepare(AlltoAllVBufferInfo &sendBuffer, AlltoAllVBufferInfo &recvBuffer,
    DeviceMem &scratchInputMem, DeviceMem &scratchOutputMem, bool isAlltoAllZCopyMode, const Stream &stream)
{
    HCCL_INFO("[AlltoAllVPairWisePro][Prepare] Begin");
    isAlltoAllZCopyMode_ = isAlltoAllZCopyMode;

    if (workMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        CHK_PRT_RET((!isAlltoAllZCopyMode_ && scratchInputMem.size() != scratchOutputMem.size()),
            HCCL_ERROR("[AlltoAllVPairWisePro][Prepare]scratchInputMem and scratchOutputMem should be the same size, "
            "ScratchInputMem[%llu] ScratchOutputMem[%llu]", scratchInputMem.size(), scratchOutputMem.size()),
            HCCL_E_MEMORY);

        CHK_PRT_RET(scratchInputMem.size() == 0 || scratchOutputMem.size() == 0,
            HCCL_ERROR("[AlltoAllVPairWisePro][Prepare] invilad scratchMemSize[%llu]", scratchInputMem.size()),
            HCCL_E_PARA);
        scratchInputMem_ = scratchInputMem;
        scratchOutputMem_ = scratchOutputMem;
        scratchMemSize_ = scratchInputMem.size();
    }

    sendBuffer_ = sendBuffer;
    recvBuffer_ = recvBuffer;
    stream_ = stream;

    CHK_RET(SalGetDataTypeSize(sendBuffer_.dataType, sendDataUnitBytes_));
    CHK_RET(SalGetDataTypeSize(recvBuffer_.dataType, recvDataUnitBytes_));

    return HCCL_SUCCESS;
}

HcclResult AlltoAllVPairWisePro::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links)
{
    HCCL_INFO("[AlltoAllVPairWisePro][RunAsync]: rank[%u] transportSize[%llu]", rank, links.size());
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());

    CHK_PRT_RET(rankSize == 0, HCCL_ERROR("[AlltoAllVPairWisePro][Prepare] invilad rankSize[%u]", rankSize), HCCL_E_PARA);

    CHK_PRT_RET(rankSize != links.size(),
        HCCL_ERROR("[AlltoAllVPairWisePro][RunAsync]: rankSize[%u] and transport size[%llu] do not match", rankSize,
        links.size()),
        HCCL_E_PARA);

    CHK_RET(LocalCopy(rank));
    if (workMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
        !isAlltoAllZCopyMode_) { // 单算子 && BCopy模式
        CHK_RET(RunBCopyAlltoAll(rank, rankSize, links));
    } else {
        CHK_RET(RunZCopyAlltoAll(rank, rankSize, links));
    }
    return HCCL_SUCCESS;
}

// 从本rank的sendbuffer拷贝到本rank的recvbuffer
HcclResult AlltoAllVPairWisePro::LocalCopy(const u32 rank)
{
    DeviceMem dstMem = recvBuffer_.mem.range(recvDataUnitBytes_ * recvBuffer_.displs[rank],
        recvBuffer_.counts[rank] * recvDataUnitBytes_);
    DeviceMem srcMem = sendBuffer_.mem.range(sendDataUnitBytes_ * sendBuffer_.displs[rank],
        sendBuffer_.counts[rank] * sendDataUnitBytes_);
    HCCL_DEBUG("[AlltoAllVPairWisePro][LocalCopy]: Rank[%u] destAddr[%p], destMax[%llu], srcAddr[%p], size[%llu]",
        rank, dstMem.ptr(), dstMem.size(), srcMem.ptr(), srcMem.size());
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, stream_));

    return HCCL_SUCCESS;
}

HcclResult AlltoAllVPairWisePro::RunBCopyAlltoAll(const u32 rank, const u32 rankSize, const std::vector<LINK> &links)
{
    for (u32 i = 1; i < rankSize; i++) {
        u32 prevRank = (rank + rankSize - i) % rankSize;
        u32 nextRank = (rank + i) % rankSize;
        std::shared_ptr<Transport> prevTransport = links[prevRank];
        std::shared_ptr<Transport> nextTransport = links[nextRank];

        CHK_SMART_PTR_NULL(prevTransport);
        CHK_SMART_PTR_NULL(nextTransport);

        HCCL_DEBUG("[AlltoAllVPairWisePro][RunBCopyAlltoAll]:prevRank[%u] nextRank[%u], step[%u]", prevRank, nextRank, i);

        u64 sendBytes = sendBuffer_.counts[nextRank] * sendDataUnitBytes_;
        u64 recvBytes = recvBuffer_.counts[prevRank] * recvDataUnitBytes_;

        u64 sendDispBytes = sendBuffer_.displs[nextRank] * sendDataUnitBytes_;
        u64 recvDispBytes = recvBuffer_.displs[prevRank] * recvDataUnitBytes_;

        // scratchMemSize_ 的合法性已经在 Prepare 函数中校验
        u32 sendTimes = (sendBytes / scratchMemSize_) + ((sendBytes % scratchMemSize_) == 0 ? 0 : 1);
        u32 recvTimes = (recvBytes / scratchMemSize_) + ((recvBytes % scratchMemSize_) == 0 ? 0 : 1);

        HCCL_DEBUG("[AlltoAllVPairWisePro][RunBCopyAlltoAll]: rank[%u] "\
                   "sendTimes[%u] recvTimes[%u] sendBytes[%llu] recvBytes[%llu] scratchMemSize_[%llu]",
                   rank, sendTimes, recvTimes, sendBytes, recvBytes, scratchMemSize_);

        u32 curSendTime = 0;
        u32 curRecvTime = 0;
        while (sendTimes != 0 || recvTimes != 0) {
            u8 *sendAddr =
                reinterpret_cast<u8 *>(sendBuffer_.mem.ptr()) + sendDispBytes + curSendTime * scratchMemSize_;
            u8 *recvAddr =
                reinterpret_cast<u8 *>(recvBuffer_.mem.ptr()) + recvDispBytes + curRecvTime * scratchMemSize_;
            u64 curSendBytes = 0;
            u64 curRecvBytes = 0;
            CHK_RET(CalcSendRecvCounts(sendTimes, curSendTime, sendBytes, curSendBytes));
            CHK_RET(CalcSendRecvCounts(recvTimes, curRecvTime, recvBytes, curRecvBytes));

            HCCL_DEBUG("[AlltoAllVPairWisePro][RunBCopyAlltoAll]: "\
                        "curSendTime[%llu] curRecvTime[%llu] curSendBytes[%llu] curRecvBytes[%llu]",
                curSendTime, curRecvTime, curSendBytes, curRecvBytes);

            HcclResult ret = SendRecv(curSendBytes, curRecvBytes, sendAddr, recvAddr, prevTransport, nextTransport);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[AlltoAllVPairWisePro][RunBCopyAlltoAll]errNo[0x%016llx] "\
                "curSendBytes[%llu] curRecvBytes[%llu] sendAddr[%p] recvAddr[%p]",
                HCCL_ERROR_CODE(ret), curSendBytes, curRecvBytes, sendAddr, recvAddr),
                ret);

            curSendTime = curSendBytes != 0 ? curSendTime + 1 : curSendTime;
            curRecvTime = curRecvBytes != 0 ? curRecvTime + 1 : curRecvTime;
            if (curSendTime == sendTimes && curRecvTime == recvTimes) {
                break;
            }
        }
    }

    return HCCL_SUCCESS;
}

HcclResult AlltoAllVPairWisePro::CalcSendRecvCounts(u32 times, u32 curTime, u64 totalBytes, u64 &curBytes) const
{
    if (times == 0) { // 不需要发送
        curBytes = 0;
    } else if (times == 1 && curTime == times - 1) { // 只发一次
        curBytes = totalBytes;
    } else if (times > 1 && totalBytes % scratchMemSize_ == 0 && curTime < times) {
        curBytes = scratchMemSize_;
    } else if (times > 1 && totalBytes % scratchMemSize_ != 0 && curTime < times - 1) {
        curBytes = scratchMemSize_;
    } else if (times > 1 && totalBytes % scratchMemSize_ != 0 && curTime == times - 1) {
        curBytes = totalBytes % scratchMemSize_;
    } else {
        curBytes = 0;
    }
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVPairWisePro::SendRecv(u64 curSendBytes, u64 curRecvBytes, u8 *sendAddr, u8 *recvAddr,
    std::shared_ptr<Transport> prevTransport, std::shared_ptr<Transport> nextTransport)
{
    if (curRecvBytes > 0) {
        CHK_RET(prevTransport->TxAck(stream_)); // transport sync record
    }
    if (curSendBytes > 0) {
        CHK_RET(nextTransport->RxAck(stream_)); // transport sync wait
        DeviceMem srcMem1 = DeviceMem::create(sendAddr, curSendBytes);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, scratchInputMem_, srcMem1, stream_));
        // send payload + notify
        CHK_RET(nextTransport->TxAsync(UserMemType::OUTPUT_MEM, 0, scratchInputMem_.ptr(), curSendBytes, stream_));
    }
    if (curRecvBytes > 0) {
        CHK_RET(prevTransport->RxAsync(UserMemType::INPUT_MEM, 0, scratchOutputMem_.ptr(), curRecvBytes, stream_));
        DeviceMem dstMem = DeviceMem::create(recvAddr, curRecvBytes);
        DeviceMem srcMem = scratchOutputMem_.range(0, curRecvBytes);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, stream_));
        CHK_RET(prevTransport->TxAck(stream_)); // record
    }
    if (curSendBytes > 0) {
        CHK_RET(nextTransport->RxAck(stream_)); // wait
        CHK_RET(nextTransport->TxDataSignal(stream_)); // record
    }
    if (curRecvBytes > 0) {
        CHK_RET(prevTransport->RxDataSignal(stream_)); // wait
        CHK_RET(prevTransport->RxWaitDone(stream_));
    }
    if (curSendBytes > 0) {
        CHK_RET(nextTransport->TxWaitDone(stream_));
    }
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVPairWisePro::SendRecv(TxMemoryInfo txMemoryInfo, RxMemoryInfo rxMemoryInfo,
    std::shared_ptr<Transport> prevTransport, std::shared_ptr<Transport> nextTransport)
{
    // send payload + notify
    CHK_RET(nextTransport->TxAsync(txMemoryInfo.dstMemType, txMemoryInfo.dstOffset, txMemoryInfo.src,
        txMemoryInfo.len, stream_));
    CHK_RET(prevTransport->RxAsync(rxMemoryInfo.srcMemType, rxMemoryInfo.srcOffset, rxMemoryInfo.dst,
        rxMemoryInfo.len, stream_));
    CHK_RET(prevTransport->TxAck(stream_)); // record
    CHK_RET(nextTransport->RxAck(stream_)); // wait
    CHK_RET(nextTransport->TxDataSignal(stream_)); // record
    CHK_RET(prevTransport->RxDataSignal(stream_)); // wait
    CHK_RET(prevTransport->RxWaitDone(stream_));
    CHK_RET(nextTransport->TxWaitDone(stream_));
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVPairWisePro::RunZCopyAlltoAll(const u32 rank, const u32 rankSize, const std::vector<LINK> &links)
{
    // 改进pairwise 总跳数为ranksize - 2
    for (u32 i = 1; i < rankSize - 1; i++) {

        if(i == 1){
            u32 prevRank_s;
            u32 nextRank_s;

            // SIO对端
            if (rank % 2 == 0){
                u32 prevRank_s = rank + 1;
                u32 nextRank_s = prevRank_s;
            }
            else{
                u32 prevRank_s = rank - 1;  
                u32 nextRank_s = prevRank_s;  
            }

            // HCCS对端
            u32 prevRank_h = (rank + rankSize - i - 1) % rankSize;
            u32 nextRank_h = (rank + i + 1) % rankSize;

            std::shared_ptr<Transport> prev_s_Transport = links[prevRank_s];
            std::shared_ptr<Transport> next_s_Transport = links[nextRank_s];

            std::shared_ptr<Transport> prev_h_Transport = links[prevRank_h];
            std::shared_ptr<Transport> next_h_Transport = links[nextRank_h];

            CHK_SMART_PTR_NULL(prev_s_Transport);
            CHK_SMART_PTR_NULL(next_s_Transport);

            CHK_SMART_PTR_NULL(prev_h_Transport);
            CHK_SMART_PTR_NULL(next_h_Transport);

            HCCL_DEBUG("[AlltoAllVPairWisePro][RunZCopyAlltoAll]: prevRank_s[%u] nextRank_s[%u], step[%u]", prevRank_s, nextRank_s, i);
            HCCL_DEBUG("[AlltoAllVPairWisePro][RunZCopyAlltoAll]: prevRank_h[%u] nextRank_h[%u], step[%u]", prevRank_h, nextRank_h, i);

            CHK_RET(prev_s_Transport->TxAck(stream_)); // transport sync record
            CHK_RET(next_s_Transport->RxAck(stream_)); // transport sync wait

            CHK_RET(prev_h_Transport->TxAck(stream_)); // transport sync record
            CHK_RET(next_h_Transport->RxAck(stream_)); // transport sync wait

            // SIO
            u64 sendBytes_s = sendBuffer_.counts[nextRank_s] * sendDataUnitBytes_;
            u64 recvBytes_s = recvBuffer_.counts[prevRank_s] * recvDataUnitBytes_;
            u64 sendDispBytes_s = sendBuffer_.displs[nextRank_s] * sendDataUnitBytes_;
            u64 recvDispBytes_s = recvBuffer_.displs[prevRank_s] * recvDataUnitBytes_;
            u8 *sendAddr_s = reinterpret_cast<u8 *>(sendBuffer_.mem.ptr()) + sendDispBytes_s;
            u8 *recvAddr_s = reinterpret_cast<u8 *>(recvBuffer_.mem.ptr()) + recvDispBytes_s;

            u64 dstOffset_s = rankRecvDisplsMap_.at(nextRank_s)[rank];
            u64 srcOffset_s = rankSendDisplsMap_.at(prevRank_s)[rank];

            TxMemoryInfo txMemoryInfo_s{UserMemType::OUTPUT_MEM, dstOffset_s, sendAddr_s, sendBytes_s};
            RxMemoryInfo rxMemoryInfo_s{UserMemType::INPUT_MEM, srcOffset_s, recvAddr_s, recvBytes_s};

            HCCL_DEBUG("[AlltoAllVPairWisePro][RunZCopyAlltoAll]: sendBytes_s[%llu] recvBytes_s[%llu] sendDispBytes_s[%llu]" \
            " dstOffset_s[%llu]", sendBytes_s, recvBytes_s, sendDispBytes_s, dstOffset_s);
            HcclResult ret = SendRecv(txMemoryInfo_s, rxMemoryInfo_s, prev_s_Transport, next_s_Transport);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[AlltoAllVPairWisePro][RunZCopyAlltoAll]errNo[0x%016llx] "\
            "sendBytes_s[%llu] recvBytes_s[%llu] sendAddr_s[%p] dstOffset_s[%llu]",
            HCCL_ERROR_CODE(ret), sendBytes_s, recvBytes_s, sendAddr_s, dstOffset_s),
            ret);

            // HCCS
            u64 sendBytes_h = sendBuffer_.counts[nextRank_h] * sendDataUnitBytes_;
            u64 recvBytes_h = recvBuffer_.counts[prevRank_h] * recvDataUnitBytes_;
            u64 sendDispBytes_h = sendBuffer_.displs[nextRank_h] * sendDataUnitBytes_;
            u64 recvDispBytes_h = recvBuffer_.displs[prevRank_h] * recvDataUnitBytes_;
            u8 *sendAddr_h = reinterpret_cast<u8 *>(sendBuffer_.mem.ptr()) + sendDispBytes_s;
            u8 *recvAddr_h = reinterpret_cast<u8 *>(recvBuffer_.mem.ptr()) + recvDispBytes_s;

            u64 dstOffset_h = rankRecvDisplsMap_.at(nextRank_h)[rank];
            u64 srcOffset_h = rankSendDisplsMap_.at(prevRank_h)[rank];

            TxMemoryInfo txMemoryInfo_h{UserMemType::OUTPUT_MEM, dstOffset_h, sendAddr_h, sendBytes_h};
            RxMemoryInfo rxMemoryInfo_h{UserMemType::INPUT_MEM, srcOffset_h, recvAddr_h, recvBytes_h};

            HCCL_DEBUG("[AlltoAllVPairWisePro][RunZCopyAlltoAll]: sendBytes_h[%llu] recvBytes_h[%llu] sendDispBytes_h[%llu]" \
            " dstOffset_h[%llu]", sendBytes_h, recvBytes_h, sendDispBytes_h, dstOffset_h);
            HcclResult ret_h = SendRecv(txMemoryInfo_h, rxMemoryInfo_h, prev_s_Transport, next_s_Transport);
            CHK_PRT_RET(ret_h != HCCL_SUCCESS,
            HCCL_ERROR("[AlltoAllVPairWisePro][RunZCopyAlltoAll]errNo[0x%016llx] "\
            "sendBytes_h[%llu] recvBytes_h[%llu] sendAddr_h[%p] dstOffset_h[%llu]",
            HCCL_ERROR_CODE(ret_h), sendBytes_h, recvBytes_h, sendAddr_h, dstOffset_h),
            ret_h);       

        }
        else{
            //最后一跳对端选择 
            u32 prevRank;
            u32 nextRank;
            if (i == rankSize - 2){
                if (rank % 2 == 0){
                    u32 prevRank = (rank - 1) % rankSize;
                    u32 nextRank = prevRank;
                }
                else{
                    u32 prevRank = (rank + 1) % rankSize;
                    u32 nextRank = prevRank;
                }
            }
            else{
                u32 prevRank = (rank + rankSize - i - 1) % rankSize;
                u32 nextRank = (rank + i + 1) % rankSize;
            }

            std::shared_ptr<Transport> prevTransport = links[prevRank];
            std::shared_ptr<Transport> nextTransport = links[nextRank];

            CHK_SMART_PTR_NULL(prevTransport);
            CHK_SMART_PTR_NULL(nextTransport);

            HCCL_DEBUG("[AlltoAllVPairWisePro][RunZCopyAlltoAll]: prevRank[%u] nextRank[%u], step[%u]", prevRank, nextRank, i);

            CHK_RET(prevTransport->TxAck(stream_)); // transport sync record
            CHK_RET(nextTransport->RxAck(stream_)); // transport sync wait

            u64 sendBytes = sendBuffer_.counts[nextRank] * sendDataUnitBytes_;
            u64 recvBytes = recvBuffer_.counts[prevRank] * recvDataUnitBytes_;
            u64 sendDispBytes = sendBuffer_.displs[nextRank] * sendDataUnitBytes_;
            u64 recvDispBytes = recvBuffer_.displs[prevRank] * recvDataUnitBytes_;
            u8 *sendAddr = reinterpret_cast<u8 *>(sendBuffer_.mem.ptr()) + sendDispBytes;
            u8 *recvAddr = reinterpret_cast<u8 *>(recvBuffer_.mem.ptr()) + recvDispBytes;

            u64 dstOffset = rankRecvDisplsMap_.at(nextRank)[rank];
            u64 srcOffset = rankSendDisplsMap_.at(prevRank)[rank];

            TxMemoryInfo txMemoryInfo{UserMemType::OUTPUT_MEM, dstOffset, sendAddr, sendBytes};
            RxMemoryInfo rxMemoryInfo{UserMemType::INPUT_MEM, srcOffset, recvAddr, recvBytes};

            HCCL_DEBUG("[AlltoAllVPairWisePro][RunZCopyAlltoAll]: sendBytes[%llu] recvBytes[%llu] sendDispBytes[%llu]"
                       " dstOffset[%llu]",
                       sendBytes, recvBytes, sendDispBytes, dstOffset);
            HcclResult ret = SendRecv(txMemoryInfo, rxMemoryInfo, prevTransport, nextTransport);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                        HCCL_ERROR("[AlltoAllVPairWisePro][RunZCopyAlltoAll]errNo[0x%016llx] "
                                   "sendBytes[%llu] recvBytes[%llu] sendAddr[%p] dstOffset[%llu]",
                                   HCCL_ERROR_CODE(ret), sendBytes, recvBytes, sendAddr, dstOffset),
                        ret);
        }
    }

    return HCCL_SUCCESS;
}
} // namespace hccl
