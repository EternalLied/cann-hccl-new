/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_batch_send_recv_retry_executor.h"
namespace hccl {
constexpr u32 PAIRSIZE_TWO = 2;

CollBatchSendRecvRetryExecutor::CollBatchSendRecvRetryExecutor(const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollBatchSendRecvExecutor(dispatcher, topoMatcher)
{
}

HcclResult CollBatchSendRecvRetryExecutor::CreatePairWiseList(HcclSendRecvItem *sendRecvInfo, u32 itemNum)
{
    HCCL_INFO("[CollBatchSendRecvRetryExecutor][GetPairWiseList] Start sort the batchSendRecv tasklist.");
    CHK_PTR_NULL(sendRecvInfo);

    for (u32 i = 0; i < itemNum; i++) {
        HCCL_INFO("[CollBatchSendRecvRetryExecutor][GetPairWiseList] index is %u, itemNum is %u, localRankID is %u, "\
            "remoteRank is %u, sendRecvType is %u, rankSize is %u.", i, itemNum, topoAttr_.userRank,
            sendRecvInfo->remoteRank, static_cast<u32>(sendRecvInfo->sendRecvType), topoAttr_.userRankSize);
        CHK_PTR_NULL(sendRecvInfo->buf);

        if (sendRecvInfo->sendRecvType == HcclSendRecvType::HCCL_SEND) {
            sendDeque_.push_back(sendRecvInfo);
        } else if (sendRecvInfo->sendRecvType == HcclSendRecvType::HCCL_RECV) {
            recvDeque_.push_back(sendRecvInfo);
        } else {
            HCCL_ERROR("[CollBatchSendRecvRetryExecutor][GetPairWiseList] sendRecvType wrong sendrecvType is %d, "\
                "rankID is %u, remoteRank is %u.", sendRecvInfo->sendRecvType, topoAttr_.userRank,
                sendRecvInfo->remoteRank);
            return HCCL_E_PARA;
        }
        sendRecvInfo++;
    }
    /* 此处的排序逻辑(pair-wise算法):
        1.sendDeque元素顺序是:先放remoteRank号小于等于root rank的第一个任务，依次减小(循环索引)直至放完
        2.recvDeque元素顺序是:先放remoteRank号大于等于root rank的第一个任务，依次增大(循环索引)直至放完
    */
    auto sendCompare = [this](HcclSendRecvItem* a, HcclSendRecvItem* b) {
        u32 aFlag = (a->remoteRank <= topoAttr_.userRank) ? (a->remoteRank + topoAttr_.userRankSize) : a->remoteRank;
        u32 bFlag = (b->remoteRank <= topoAttr_.userRank) ? (b->remoteRank + topoAttr_.userRankSize) : b->remoteRank;
        return aFlag > bFlag;
    };

    auto recvCompare = [this](HcclSendRecvItem* a, HcclSendRecvItem* b) {
        u32 aFlag = (a->remoteRank < topoAttr_.userRank) ? (a->remoteRank + topoAttr_.userRankSize) : a->remoteRank;
        u32 bFlag = (b->remoteRank < topoAttr_.userRank) ? (b->remoteRank + topoAttr_.userRankSize) : b->remoteRank;
        return aFlag < bFlag;
    };

    std::sort(sendDeque_.begin(), sendDeque_.end(), sendCompare);
    std::sort(recvDeque_.begin(), recvDeque_.end(), recvCompare);

    // 生成SendRecvPair
    u32 pairNum = std::max(sendDeque_.size(), recvDeque_.size());
    for (u32 pairIndex = 0; pairIndex < pairNum; pairIndex++) {
        std::vector<HcclSendRecvItem*> sendRecvPair;
        if (sendDeque_.size() > pairIndex) {
           sendRecvPair.push_back(sendDeque_[pairIndex]);
        }
        if (recvDeque_.size() > pairIndex) {
            sendRecvPair.push_back(recvDeque_[pairIndex]);
        }
        sendRecvPairList_.push_back(sendRecvPair);
    }
    HCCL_INFO("[CollBatchSendRecvRetryExecutor][GetPairWiseList] End sort the batchSendRecv tasklist.");
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvRetryExecutor::GetPairWiseList(std::vector<std::vector<HcclSendRecvItem*>> &sendRecvPairList)
{
    sendRecvPairList = sendRecvPairList_;
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvRetryExecutor::CheckSendRecvPair(const std::vector<HcclSendRecvItem*> &sendRecvPair)
{
    if (sendRecvPair.empty()) {
        HCCL_ERROR("[CollBatchSendRecvRetryExecutor] please check the pair list.");
        return HCCL_E_PARA;
    }
    if (sendRecvPair.size() == 1 && sendRecvPair[0]->remoteRank == topoAttr_.userRank) {
        HCCL_ERROR("[CollBatchSendRecvRetryExecutor] SendTask and Recv Task to rank itself do not match,"\
            "please check the task list.");
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvRetryExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algResource)
{
    HcclUs startut = TIME_NOW();
    algResResp_ = &algResource;

    HCCL_PROFILER_ADD_TAG(param.tag, algoAttr_.identifier, workflowMode_);
    HCCL_PROFILER_ADD_STREAM_BY_STREAMID(param.stream.id(), param.tag, 0, algType_);
    CHK_RET(AddSubStreamToProfiling());

    // 校验当前sendRecvPair
    std::vector<HcclSendRecvItem*> sendRecvPair;
    if (param.BatchSendRecvDataDes.curIterNum < sendRecvPairList_.size()) {
        sendRecvPair = sendRecvPairList_[param.BatchSendRecvDataDes.curIterNum];
    } else {
        HCCL_ERROR("[CollBatchSendRecvRetryExecutor] the curIterNum[%u] is out of range[0, %zu].",
            param.BatchSendRecvDataDes.curIterNum, sendRecvPairList_.size());
        return HCCL_E_PARA;
    }
    CHK_RET(CheckSendRecvPair(sendRecvPair));

    // 自发自收场景
    if (sendRecvPair.size() == PAIRSIZE_TWO && sendRecvPair[0]->remoteRank == topoAttr_.userRank &&
        sendRecvPair[1]->remoteRank == topoAttr_.userRank) {
        if (sendRecvPair[0]->count == sendRecvPair[1]->count && sendRecvPair[0]->dataType == sendRecvPair[1]->dataType) {
            u64 dataSize = sendRecvPair[0]->count * SIZE_TABLE[sendRecvPair[0]->dataType];
            DeviceMem inUserMem = DeviceMem::create(static_cast<u8*>(sendRecvPair[0]->buf), dataSize);
            DeviceMem outUserMem = DeviceMem::create(static_cast<u8*>(sendRecvPair[1]->buf), dataSize);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, outUserMem, inUserMem, param.stream));
            return HCCL_SUCCESS;
        } else {
             HCCL_ERROR("[HcclBatchSendRecvRetry] Send task and recv task to self : data size do not equal, please"\
                "check the task list.");
            return HCCL_E_PARA;
        }
    }

    // 重执行正常执行场景，前后需和控制流做同步
    if (param.BatchSendRecvDataDes.curMode == BatchSendRecvCurMode::SEND_RECV) {
        HCCL_INFO("[BatchSendRecv] Stream sync: main stream record, subStream wait.");
        CHK_RET(LocalNotify::Post(param.stream, dispatcher_, algResResp_->notifiesS2M[STREAM_INDEX_0], PROF_STAGE_0));
        CHK_RET(LocalNotify::Wait(algResResp_->slaveStreams[STREAM_INDEX_0], dispatcher_,
            algResResp_->notifiesS2M[STREAM_INDEX_0], PROF_STAGE_0));
        CHK_RET(LocalNotify::Post(param.stream, dispatcher_, algResResp_->notifiesS2M[STREAM_INDEX_1], PROF_STAGE_1));
        CHK_RET(LocalNotify::Wait(algResResp_->slaveStreams[STREAM_INDEX_1], dispatcher_,
            algResResp_->notifiesS2M[STREAM_INDEX_1], PROF_STAGE_1));
    }
    // run sendrecv
    CHK_RET(RunLoop(param, algResource, sendRecvPair));

    if (param.BatchSendRecvDataDes.curMode == BatchSendRecvCurMode::SEND_RECV) {
        HCCL_INFO("[BatchSendRecv] Stream sync: subStream record, main stream wait.");
        CHK_RET(LocalNotify::Post(algResResp_->slaveStreams[STREAM_INDEX_0], dispatcher_,
            algResResp_->notifiesM2S[STREAM_INDEX_0], PROF_STAGE_0));
        CHK_RET(LocalNotify::Wait(param.stream, dispatcher_, algResResp_->notifiesM2S[STREAM_INDEX_0],
            PROF_STAGE_0));
        CHK_RET(LocalNotify::Post(algResResp_->slaveStreams[STREAM_INDEX_1], dispatcher_,
            algResResp_->notifiesM2S[STREAM_INDEX_1], PROF_STAGE_1));
        CHK_RET(LocalNotify::Wait(param.stream, dispatcher_, algResResp_->notifiesM2S[STREAM_INDEX_1],
            PROF_STAGE_1));
    } else if (param.BatchSendRecvDataDes.curMode == BatchSendRecvCurMode::SEND) {
        CHK_RET(LocalNotify::Post(algResResp_->slaveStreams[STREAM_INDEX_0], dispatcher_,
            algResResp_->notifiesM2S[STREAM_INDEX_0], PROF_STAGE_0));
    } else if (param.BatchSendRecvDataDes.curMode == BatchSendRecvCurMode::RECV) {
        CHK_RET(LocalNotify::Post(algResResp_->slaveStreams[STREAM_INDEX_1], dispatcher_,
            algResResp_->notifiesM2S[STREAM_INDEX_1], PROF_STAGE_1));
    }

    CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));
    HCCL_INFO("[debug][print] LaunchTaskExtend success.");
    HCCL_PROFILER_DEL_STREAM_BY_STREAMID(param.stream.id());
    HCCL_PROFILER_DEL_TAG(param.tag);
    HCCL_INFO("tag[%s] BatchSendRecv Excutor orchestrate success, take time [%lld]us.",
        param.tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvRetryExecutor::RunLoop(OpParam &param, AlgResourceResponse &algRes,
    const std::vector<HcclSendRecvItem*> &sendRecvPair)
{
    HcclResult ret = HCCL_SUCCESS;
    // 判断当前需执行的算子
    std::vector<HcclSendRecvItem*> curSendRecvPair;
    if (param.BatchSendRecvDataDes.curMode == BatchSendRecvCurMode::SEND) {
        curSendRecvPair.push_back(sendRecvPair[0]);
    } else if (param.BatchSendRecvDataDes.curMode == BatchSendRecvCurMode::RECV) {
        curSendRecvPair.push_back(sendRecvPair[sendRecvPair.size() - 1]);
    } else {
        curSendRecvPair = sendRecvPair;
    }
    // 执行当前需执行的算子
    for (u32 opIndex = 0; opIndex < curSendRecvPair.size(); opIndex++) {
        u32 unitSize = SIZE_TABLE[curSendRecvPair[opIndex]->dataType];
        u8 *curInputPtr = nullptr;
        u8 *curOutputPtr = nullptr;
        u64 maxCountPerLoop = 0U;
        if (curSendRecvPair[opIndex]->sendRecvType == HcclSendRecvType::HCCL_SEND) {
            curInputPtr = static_cast<u8 *>(curSendRecvPair[opIndex]->buf);
            CHK_PTR_NULL(curInputPtr);
            maxCountPerLoop = CalcSendLoopMaxCount(const_cast<DeviceMem&>(algRes.cclInputMem), unitSize);
        } else if (curSendRecvPair[opIndex]->sendRecvType == HcclSendRecvType::HCCL_RECV) {
            curOutputPtr = static_cast<u8 *>(curSendRecvPair[opIndex]->buf);
            CHK_PTR_NULL(curOutputPtr);
            maxCountPerLoop = CalcRecvLoopMaxCount(const_cast<DeviceMem&>(algRes.cclOutputMem), unitSize);
        } else {
            HCCL_ERROR("[CollBatchSendRecvRetryExecutor][RunLoop] sendRecvType is Wrong.");
            return HCCL_E_PARA;
        }
        CHK_RET(GetSendRecvInfo(curSendRecvPair[opIndex]));
        for (u64 countLeft = curSendRecvPair[opIndex]->count, curCount = 0, curOffset = 0; countLeft > 0;
            countLeft -= curCount) {
            curInputPtr += curOffset;
            curOutputPtr += curOffset;
            curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
            u64 curSize = curCount * unitSize; // 单位：字节

            if (curSendRecvPair[opIndex]->sendRecvType == HcclSendRecvType::HCCL_SEND) {
                DeviceMem inMem(curInputPtr, curSize);
                DeviceMem inCommMem = algRes.cclInputMem.range(0, curSize);
                CHK_RET(HcclD2DMemcpyAsync(dispatcher_, inCommMem, inMem, algResResp_->slaveStreams[STREAM_INDEX_0]));
            }
            ExecMem execMem;
            execMem.inputMem = algRes.cclInputMem.range(0, curSize);
            execMem.outputMem = algRes.cclOutputMem.range(0, curSize);
            ret = KernelRun(param, execMem);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollBatchSendRecvRetryExecutor][RunLoop]errNo[0x%016llx]kernel run error, tag[%s], " \
                "sendRecvType[%d], input_ptr[%p], output_ptr[%p], count[%llu], data_type[%s]",
                HCCL_ERROR_CODE(ret), param.tag.c_str(), sendRecvType_, execMem.inputMem.ptr(), execMem.outputMem.ptr(),
                curCount, GetDataTypeEnumStr(curSendRecvPair[opIndex]->dataType).c_str()), ret);
            if (curSendRecvPair[opIndex]->sendRecvType == HcclSendRecvType::HCCL_RECV) {
                DeviceMem outMem(curOutputPtr, curSize);
                DeviceMem outCommMem = algRes.cclOutputMem.range(0, curSize);
                CHK_RET(HcclD2DMemcpyAsync(dispatcher_, outMem, outCommMem, algResResp_->slaveStreams[STREAM_INDEX_1]));
            }
            CHK_PRT_RET((curCount == 0), HCCL_ERROR("[Loop][BatchSendRecv] In OP_BASE curCount is zero."), HCCL_E_PARA);
            curOffset = curSize;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvRetryExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    CHK_RET(CheckCommSize(COMM_COMBINE_ORDER, COMM_SIZE_TWO));
    u32 commIndex = 0;
    HCCL_INFO("[CollBatchSendRecvRetryExecutor][KernelRun] sendRecvType_[%d], remoteUserRank_[%u], userRank_[%u].",
        sendRecvType_, remoteUserRank_, topoAttr_.userRank);
    if ((sendRecvType_ == HcclSendRecvType::HCCL_SEND && remoteUserRank_ < topoAttr_.userRank) ||
        (sendRecvType_ == HcclSendRecvType::HCCL_RECV && remoteUserRank_ > topoAttr_.userRank)) {
        HCCL_INFO("[CollBatchSendRecvRetryExecutor][KernelRun] CommIndex is 0.");
        commIndex = COMM_INDEX_0;
    } else if ((sendRecvType_ == HcclSendRecvType::HCCL_SEND && remoteUserRank_ > topoAttr_.userRank) ||
                (sendRecvType_ == HcclSendRecvType::HCCL_RECV && remoteUserRank_ < topoAttr_.userRank)) {
        HCCL_INFO("[CollBatchSendRecvRetryExecutor][KernelRun] CommIndex is 1.");
        commIndex = COMM_INDEX_1;
    } else {
        HCCL_ERROR("[CollBatchSendRecvRetryExecutor][KernelRun] CommIndex doesn't match.");
        return HCCL_E_INTERNAL;
    }
    CHK_PRT_RET(commIndex >= algResResp_->opTransportResponse[COMM_COMBINE_ORDER].size(),
        HCCL_ERROR("[CollBatchSendRecvRetryExecutor][KernelRun] batchsendrecv op commIndex[%u] is larger than "\
        "opTransportResponse size[%zu]",
        remoteUserRank_, algResResp_->opTransportResponse[COMM_COMBINE_ORDER].size()), HCCL_E_NOT_SUPPORT);
    SingleSubCommTransport &commCombined =
        const_cast<SingleSubCommTransport&>(algResResp_->opTransportResponse[COMM_COMBINE_ORDER][commIndex]);

    CHK_PRT_RET(remoteUserRank_ >= commCombined.userRank2subCommRank.size(),
        HCCL_ERROR("[CollBatchSendRecvRetryExecutor][KernelRun] batchsendrecv op remoteUserRank[%u] is larger than "\
        "userRank2subCommRank map size[%zu]",
        remoteUserRank_, commCombined.userRank2subCommRank.size()), HCCL_E_NOT_SUPPORT);

    u32 remoteRank = commCombined.userRank2subCommRank[remoteUserRank_];
    CHK_PRT_RET(remoteRank >= commCombined.links.size(),
        HCCL_ERROR("[CollBatchSendRecvRetryExecutor][KernelRun] batchsendrecv op remoteUserRank[%u], get remoteRank[%u]," \
        "the size of combinedCcomm links is [%zu]", remoteUserRank_, remoteRank, commCombined.links.size()),
        HCCL_E_NOT_SUPPORT);
    LINK &targetLink = commCombined.links[remoteRank];
    CHK_SMART_PTR_NULL(targetLink);
    SendReceive executor(dispatcher_, targetLink);

    if (sendRecvType_ == HcclSendRecvType::HCCL_SEND) {
        CHK_RET(executor.SendPrepare(execMem.inputMem, remoteUserRank_, algResResp_->slaveStreams[STREAM_INDEX_0]));
        CHK_RET(executor.RegisterProfiler(0, PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, algResResp_->slaveStreams[STREAM_INDEX_0]));
        CHK_RET(executor.BatchSendRunAsync());
    } else if (sendRecvType_ == HcclSendRecvType::HCCL_RECV) {
        CHK_RET(executor.ReceivePrepare(execMem.outputMem, remoteUserRank_, algResResp_->slaveStreams[STREAM_INDEX_1]));
        CHK_RET(executor.RegisterProfiler(0, PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET,
            algResResp_->slaveStreams[STREAM_INDEX_1]));
        CHK_RET(executor.BatchReceiveRunAsync());
    } else {
        HCCL_ERROR("[CollBatchSendRecvRetryExecutor][KernelRun] SendRecvType doesn't match. RemoteRank is [%u]",
            remoteUserRank_);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvRetryExecutor::CalcStreamNum(u32& streamNum)
{
    streamNum = OUTER_PLANE_NUM_IN_NPRING_DOUBLE;
    HCCL_INFO("[CollBatchSendRecvRetryExecutor][CalcScratchMemSize] tag_[%s], streamNum[%u].", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

REGISTER_EXEC("BatchSendRecvRetry", BatchSendRecvRetryExecutor, CollBatchSendRecvRetryExecutor);
} // namespace hccl