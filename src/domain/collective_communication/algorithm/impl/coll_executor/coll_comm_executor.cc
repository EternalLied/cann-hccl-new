/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_comm_executor.h"
#include "executor_impl.h"
#include "stream_active_manager.h"
#include "device_capacity.h"
#include "comm_factory_pub.h"
#include "externalinput_pub.h"

namespace hccl {
CollCommExecutor::CollCommExecutor(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollNativeExecutorBase(dispatcher, topoMatcher)
{
}

HcclResult CollCommExecutor::GetSubStreamInfoOnOneRing(const u32 ringIndex,
                                                       std::vector<Stream>                       &subStreamsInOneRing,
                                                       std::vector<std::shared_ptr<LocalNotify>> &mainSignalsInOneRing,
                                                       std::vector<std::shared_ptr<LocalNotify>> &subSignalsInOneRing)
{
    u32 ringNum = algResResp_->slaveStreams.size() + 1;
    if (ringNum == OUTER_PLANE_NUM_IN_NPRING_DOUBLE * STREAM_NUM_FOR_DMAREDUCE_ONE_RING) {
        // double ring
        subStreamsInOneRing.push_back(algResResp_->slaveStreams[ringIndex + 1]);
        mainSignalsInOneRing.push_back(algResResp_->notifiesM2S[ringIndex + 1]);
        subSignalsInOneRing.push_back(algResResp_->notifiesS2M[ringIndex + 1]);
    } else if (ringNum == OUTER_PLANE_NUM_IN_NPRING_SINGLE * STREAM_NUM_FOR_DMAREDUCE_ONE_RING) {
        // single ring
        subStreamsInOneRing.push_back(algResResp_->slaveStreams[ringIndex]);
        mainSignalsInOneRing.push_back(algResResp_->notifiesM2S[ringIndex]);
        subSignalsInOneRing.push_back(algResResp_->notifiesS2M[ringIndex]);
    }
    return HCCL_SUCCESS;
}

HcclResult CollCommExecutor::MultiRingAllReduce(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
    const u64 count, const HcclDataType dataType, const HcclReduceOp reductionOp,
    const std::vector<std::vector<Slice>> &multRingsSliceZero, Stream stream, s32 profStage,
    const u64 baseOffset)
{
    HcclResult ret = HCCL_SUCCESS;
    u32 ringNum = multRingsSliceZero.size();
    CHK_RET(CheckCommSize(COMM_LEVEL0, ringNum));

    u64 reduceAttr = GetReduceAttr(inputMem, outputMem, dataType, reductionOp);

    std::vector<std::vector<u32>> ringNics;
    CHK_RET(GetRingNics(tag, ringNics));

    for (u32 ringIndex = 0; ringIndex < ringNum; ringIndex++) {
        std::vector<Slice> singleRingSliceZero = multRingsSliceZero[ringIndex];
        CHK_PRT_RET(singleRingSliceZero.empty(),
            HCCL_ERROR("[CollCommExecutor][MultiRingAllReduce]singleRingSliceZero is empty"), HCCL_E_INTERNAL);

        SubCommInfo outerRingCommInfo = GetSubCommInfo(COMM_LEVEL0, ringIndex);

        u32 rankSize = outerRingCommInfo.localRankSize;
        u32 ringIndexOp = ringIndex;
        std::unique_ptr<ExecutorBase> executor;
        executor.reset(new (std::nothrow) AllReduceRing(dispatcher_, reduceAttr));
        CHK_SMART_PTR_NULL(executor);

        if (ringIndex != (ringNum - 1)) {  // 0~ringNum-2的环
            if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) { // offline
                CHK_RET(StreamActiveManager::GetInstance(topoAttr_.deviceLogicId).StreamActive(
                    algResResp_->slaveStreams[ringIndex].ptr(), stream.ptr()));
            }

            ret = LocalNotify::Wait(algResResp_->slaveStreams[ringIndex], dispatcher_,
                algResResp_->notifiesS2M[ringIndex], profStage);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[CollCommExecutor][MultiRingAllReduce]stream[%u] wait failed",
                ringIndex), ret);
            ret = executor->Prepare(inputMem, outputMem, outputMem, count, dataType,
                algResResp_->slaveStreams[ringIndex], reductionOp, OUTER_BRIDGE_RANK_ID, singleRingSliceZero,
                baseOffset, ringNics[ringIndex]);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingAllReduce]stream[%u], allreduce(ring) prepare failed,"\
                "return[%d]", ringIndex, ret), ret);

            ret = executor->RegisterProfiler(
                ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + outerRingCommInfo.localRank,
                profStage, HCCL_EXEC_STEP_NOT_SET, algResResp_->slaveStreams[ringIndex]);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingAllReduce]stream[%u], allreduce(ring) register Profiler "\
                "failed,return[%d]", ringIndex, ret), ret);

            ret = RunTemplate(executor, outerRingCommInfo);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingAllReduce]stream[%u], allreduce(ring) run failed,"\
                "return[%d]", ringIndex, ret), ret);

            ret = LocalNotify::Post(algResResp_->slaveStreams[ringIndex], dispatcher_, algResResp_->notifiesM2S[ringIndex],
                profStage);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingAllReduce]stream[%u] record failed", ringIndex), ret);

            ret = LocalNotify::Post(stream, dispatcher_, algResResp_->notifiesS2M[ringIndex], profStage);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingAllReduce]stream[%u] record failed", ringIndex), ret);
        } else {  // 主环
            executor.reset(new (std::nothrow) AllReduceRing(dispatcher_, reduceAttr));
            CHK_SMART_PTR_NULL(executor);
            ret = executor->Prepare(inputMem, outputMem, outputMem, count, dataType, stream,
                reductionOp, OUTER_BRIDGE_RANK_ID, singleRingSliceZero, baseOffset, ringNics[ringIndex]);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingAllReduce]stream[%u], allreduce(ring) prepare failed, "\
                "return[%d]", ringIndex, ret), ret);

            ret = executor->RegisterProfiler(
                ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + outerRingCommInfo.localRank,
                profStage, HCCL_EXEC_STEP_NOT_SET, stream);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingAllReduce]stream[%u], allreduce(ring) register Profiler "\
                "failed,return[%d]", ringIndex, ret), ret);

            ret = RunTemplate(executor, outerRingCommInfo);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingAllReduce]stream[%u], allreduce(ring) run failed, "\
                "return[%d]", ringIndex, ret), ret);

            for (u32 ring = 0; ring < (ringNum - 1); ring++) {
                /* 等待executor执行完毕 */
            ret = LocalNotify::Wait(stream, dispatcher_, algResResp_->notifiesM2S[ring], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllReduce]stream[%u] wait failed", ring), ret);
            }
        }
    }

    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}

HcclResult CollCommExecutor::MultiRingAllGather(const std::string &tag, DeviceMem inputMem, DeviceMem outputMem,
    const u64 count, const HcclDataType dataType, const std::vector<std::vector<Slice> > multRingsSliceZero,
    Stream stream, s32 profStage, const u64 baseOffset, const HcomCollOpInfo *opInfo,
    const std::vector<std::vector<Slice>> multRingsUserMemSlice)
{
    HcclResult ret = HCCL_SUCCESS;
    u32 ringNum = multRingsSliceZero.size();
    CHK_RET(CheckCommSize(COMM_LEVEL0, ringNum));

    std::vector<std::vector<u32>> ringNics;
    CHK_RET(GetRingNics(tag, ringNics));
    // 拿到ring环映射关系
    SubCommInfo outerZeroCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    auto nicList = topoAttr_.nicList;
    std::vector<std::vector<u32>> multiRingsOrder =
        GetRingsOrderByTopoType(outerZeroCommInfo.localRankSize, topoType_, nicList);

    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    for (u32 ringIndex = 0; ringIndex < ringNum; ringIndex++) {
        std::vector<Slice> singleRingSliceZero = multRingsSliceZero[ringIndex];
        CHK_PRT_RET(singleRingSliceZero.empty(), HCCL_ERROR("[CollCommExecutor][MultiRingAllGather]"\
            "singleRingSliceZero is empty"), HCCL_E_INTERNAL);

        // 910_93场景 生成userMemOut_上对应的slices
        std::vector<Slice> userMemOutputSlices;
        if (multRingsUserMemSlice.size() == 0) {
            CHK_RET(CalUserMemSlices(dataType, opInfo, singleRingSliceZero, ringIndex, multiRingsOrder,
                userMemOutputSlices));
        } else {
            userMemOutputSlices = multRingsUserMemSlice[ringIndex];
        }
        std::vector<u32> rankOrder;
        CHK_RET(GetRankOrder(multiRingsOrder, ringIndex, rankOrder));

        SubCommInfo outerRingCommInfo = GetSubCommInfo(COMM_LEVEL0, ringIndex);

        u32 rankSize = outerRingCommInfo.localRankSize;
        u32 ringIndexOp = ringIndex;

        // 910_93场景 准备环中的从流
        std::vector<Stream>       subStreamsInOneRing;
        std::vector<std::shared_ptr<LocalNotify>> mainSignalsInOneRing;
        std::vector<std::shared_ptr<LocalNotify>> subSignalsInOneRing;
        if (opInfo != nullptr) {
            CHK_RET(GetSubStreamInfoOnOneRing(ringIndex, subStreamsInOneRing, mainSignalsInOneRing,
                                              subSignalsInOneRing));
        }
        if (ringIndex != (ringNum - 1)) { // 最后一个环是主stream，所以这里减1，符合条件的走从stream
            if (!topoMatcher_->GetExternalInputHcclEnableFfts() &&
                workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                if (opInfo != nullptr) {
                    algResResp_->threadManage[ringIndex]->Prepare(
                        outputMem, outputMem, inputMem, count, dataType,
                        algResResp_->slaveStreams[ringIndex], HcclReduceOp::HCCL_REDUCE_RESERVED, OUTER_BRIDGE_RANK_ID,
                        singleRingSliceZero, baseOffset, ringNics[ringIndex], tag, profStage,
                        outerRingCommInfo, algResResp_->notifiesS2M[ringIndex], algResResp_->notifiesM2S[ringIndex],
                        ringIndex, ExecutorType::ALLGATHER_RING_DIRECT, 0, opInfo, subStreamsInOneRing,
                        mainSignalsInOneRing, subSignalsInOneRing, rankOrder, userMemOutputSlices);
                } else {
                    algResResp_->threadManage[ringIndex]->Prepare(outputMem, outputMem, inputMem, count, dataType,
                        algResResp_->slaveStreams[ringIndex], HcclReduceOp::HCCL_REDUCE_RESERVED, OUTER_BRIDGE_RANK_ID,
                        singleRingSliceZero, baseOffset, ringNics[ringIndex], tag, profStage,
                        outerRingCommInfo, algResResp_->notifiesS2M[ringIndex], algResResp_->notifiesM2S[ringIndex],
                        ringIndex, ExecutorType::ALLGATHER_RING);
                }
                algResResp_->threadManage[ringIndex]->NotifyStart();    // 给线程发信号启动处理
            } else {
                ret = LocalNotify::Wait(algResResp_->slaveStreams[ringIndex], dispatcher_,
                    algResResp_->notifiesS2M[ringIndex], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllGather]stream[%u] wait failed", ringIndex), ret);
                // 如何判断是否环内是否有数据, 以ring的第一个rank的 size为判断依据
                std::unique_ptr<ExecutorBase> executor;
                if (opInfo != nullptr) {
                    executor.reset(new (std::nothrow) AllGatherRingConcurrentDirect(
                        dispatcher_, opInfo, topoAttr_.userRank, subStreamsInOneRing,
                        mainSignalsInOneRing, subSignalsInOneRing, rankOrder, userMemOutputSlices));
                } else {
                    executor.reset(new (std::nothrow) AllGatherRing(dispatcher_));
                }
                CHK_SMART_PTR_NULL(executor);
                ret = executor->Prepare(outputMem, outputMem, inputMem, count, dataType,
                    algResResp_->slaveStreams[ringIndex], HcclReduceOp::HCCL_REDUCE_RESERVED, OUTER_BRIDGE_RANK_ID,
                    singleRingSliceZero, baseOffset, ringNics[ringIndex]);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllGather]stream[%u],all gather(ring) prepare "\
                    "failed,return[%d]", ringIndex, ret), ret);
                ret = executor->RegisterProfiler(
                    ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                    (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + outerRingCommInfo.localRank,
                    profStage, HCCL_EXEC_STEP_NOT_SET, algResResp_->slaveStreams[ringIndex]);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllGather]stream[%u],all gather(ring) register "\
                    "Profiler failed,return[%d]", ringIndex, ret), ret);

                ret = RunTemplate(executor, outerRingCommInfo);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllGather]stream[%u],all gather(ring) run failed, "\
                    "return[%d]", ringIndex, ret), ret);

                ret = LocalNotify::Post(algResResp_->slaveStreams[ringIndex], dispatcher_,
                    algResResp_->notifiesM2S[ringIndex], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllGather]stream[%u] record failed",
                    ringIndex), ret);
            }

            ret = LocalNotify::Post(stream, dispatcher_, algResResp_->notifiesS2M[ringIndex], profStage);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingAllGather]stream[%u] record failed", ringIndex), ret);
        } else { // 主环
            std::unique_ptr<ExecutorBase> executor;
            if (opInfo != nullptr) {
                executor.reset(new (std::nothrow) AllGatherRingConcurrentDirect(
                    dispatcher_, opInfo, topoAttr_.userRank, subStreamsInOneRing, mainSignalsInOneRing,
                    subSignalsInOneRing, rankOrder, userMemOutputSlices));
            } else {
                executor.reset(new (std::nothrow) AllGatherRing(dispatcher_));
            }
            CHK_SMART_PTR_NULL(executor);
            ret = executor->Prepare(outputMem, outputMem, inputMem, count, dataType, stream, HCCL_REDUCE_RESERVED,
                OUTER_BRIDGE_RANK_ID, singleRingSliceZero, baseOffset, ringNics[ringIndex]);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingAllGather]stream[%u],all gather(ring) prepare failed,"\
                "return[%d]", ringIndex, ret), ret);

            ret = executor->RegisterProfiler(
                ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + outerRingCommInfo.localRank,
                profStage, HCCL_EXEC_STEP_NOT_SET, stream);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingAllGather]stream[%u], all gather(ring) register Profiler "\
                "failed,return[%d]", ringIndex, ret), ret);

            ret = RunTemplate(executor, outerRingCommInfo);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingAllGather]stream[%u], all gather(ring) run failed,"\
                "return[%d]", ringIndex, ret), ret);

            for (u32 ring = 0; ring < (ringNum - 1); ring++) {
                if (!topoMatcher_->GetExternalInputHcclEnableFfts() &&
                    workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                    algResResp_->threadManage[ring]->WaitDone(); // 单算子模式，等待线程处理完成信号
                }
                ret = LocalNotify::Wait(stream, dispatcher_, algResResp_->notifiesM2S[ring], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllGather]stream[%u] wait failed", ring), ret);
            }
        }
    }

    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}

HcclResult CollCommExecutor::MultiRingAllGatherConcurrent(const std::string &tag, DeviceMem inputMem,
    DeviceMem outputMem, const u64 count, const HcclDataType dataType,
    const std::vector<std::pair<bool, std::vector<Slice>>> multRingsSliceZero,
    Stream stream, s32 profStage, const u64 baseOffset, const HcomCollOpInfo *opInfo,
    const std::vector<std::pair<bool, std::vector<Slice>>> multRingsUserMemSlice)
{
    HcclResult ret = HCCL_SUCCESS;
    u32 ringNum = multRingsSliceZero.size(); // 环数, 当前为4环

    std::vector<std::vector<u32>> ringNics;
    CHK_RET(GetRingNics(tag, ringNics));
    auto halfRingSize = ringNum;
    if (ringNum > RDMA_PLANE_NUM_IN_NPRING_DOUBLE) {
        halfRingSize = ringNum / 2; // 2环
    }
    // 拿到ring环映射关系
    SubCommInfo outerZeroCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    auto nicList = topoAttr_.nicList;
    std::vector<std::vector<u32>> multiRingsOrder =
        GetRingsOrderByTopoType(outerZeroCommInfo.localRankSize, topoType_, nicList);

    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    for (u32 ringIndex = 0; ringIndex < ringNum; ringIndex++) {
        std::vector<Slice> singleRingSliceZero = multRingsSliceZero[ringIndex].second; // 取出sdma/rdma的数据块
        CHK_PRT_RET(singleRingSliceZero.empty(), HCCL_ERROR("[CollCommExecutor][MultiRingAllGatherConcurrent]"\
            "singleRingSliceZero is empty"), HCCL_E_INTERNAL);

        // 910_93场景 生成userMemOut_上对应的slices
        std::vector<Slice> userMemOutputSlices;
        if (multRingsUserMemSlice.size() == 0) {
            CHK_RET(CalUserMemSlices(dataType, opInfo, singleRingSliceZero, ringIndex, multiRingsOrder,
                userMemOutputSlices));
        } else {
            userMemOutputSlices = multRingsUserMemSlice[ringIndex].second;
        }
        std::vector<u32> rankOrder;
        u32 commIndex = ringIndex % halfRingSize;
        CHK_RET(GetRankOrder(multiRingsOrder, commIndex, rankOrder));

        SubCommInfo outerRingCommInfo = multRingsSliceZero[ringIndex].first ?
            GetSubCommInfo(COMM_LEVEL0, commIndex) : GetSubCommInfo(COMM_LEVEL0_RDMA, commIndex);

        u32 rankSize = outerRingCommInfo.localRankSize;
        u32 ringIndexOp = ringIndex;

        // 910_93场景 准备环中的从流
        std::vector<Stream>       subStreamsInOneRing;
        std::vector<std::shared_ptr<LocalNotify>> mainSignalsInOneRing;
        std::vector<std::shared_ptr<LocalNotify>> subSignalsInOneRing;
        if (opInfo != nullptr) {
            CHK_RET(GetSubStreamInfoOnOneRing(ringIndex, subStreamsInOneRing, mainSignalsInOneRing,
                                              subSignalsInOneRing));
        }
        bool isSdma = multRingsSliceZero[ringIndex].first;
        if (ringIndex != (ringNum - 1)) { // 最后一个环是主stream，所以这里减1，符合条件的走从stream
            if (!topoMatcher_->GetExternalInputHcclEnableFfts() &&
                workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                if (opInfo != nullptr) {
                    ExecutorType type = isSdma ?
                        ExecutorType::ALLGATHER_RING_DIRECT : ExecutorType::ALLGATHER_RING_DIRECT_RDMA;
                    algResResp_->threadManage[ringIndex]->Prepare(
                        outputMem, outputMem, inputMem, count, dataType,
                        algResResp_->slaveStreams[ringIndex], HcclReduceOp::HCCL_REDUCE_RESERVED, OUTER_BRIDGE_RANK_ID,
                        singleRingSliceZero, baseOffset, ringNics[ringIndex%halfRingSize], tag, profStage,
                        outerRingCommInfo, algResResp_->notifiesS2M[ringIndex], algResResp_->notifiesM2S[ringIndex],
                        ringIndex, type, 0, opInfo, subStreamsInOneRing,
                        mainSignalsInOneRing, subSignalsInOneRing, rankOrder, userMemOutputSlices);
                } else {
                    algResResp_->threadManage[ringIndex]->Prepare(outputMem, outputMem, inputMem, count, dataType,
                        algResResp_->slaveStreams[ringIndex], HcclReduceOp::HCCL_REDUCE_RESERVED, OUTER_BRIDGE_RANK_ID,
                        singleRingSliceZero, baseOffset, ringNics[ringIndex%halfRingSize], tag, profStage,
                        outerRingCommInfo, algResResp_->notifiesS2M[ringIndex], algResResp_->notifiesM2S[ringIndex],
                        ringIndex, ExecutorType::ALLGATHER_RING);
                }
                algResResp_->threadManage[ringIndex]->NotifyStart();    // 给线程发信号启动处理
            } else {
                ret = LocalNotify::Wait(algResResp_->slaveStreams[ringIndex], dispatcher_,
                    algResResp_->notifiesS2M[ringIndex], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u] wait failed",
                    ringIndex), ret);
                // 如何判断是否环内是否有数据, 以ring的第一个rank的 size为判断依据
                std::unique_ptr<ExecutorBase> executor;
                if (opInfo != nullptr) {
                    executor.reset(new (std::nothrow) AllGatherRingConcurrentDirect(
                        dispatcher_, opInfo, topoAttr_.userRank, subStreamsInOneRing,
                        mainSignalsInOneRing, subSignalsInOneRing, rankOrder, userMemOutputSlices, isSdma));
                } else {
                    executor.reset(new (std::nothrow) AllGatherRing(dispatcher_));
                }
                CHK_SMART_PTR_NULL(executor);
                ret = executor->Prepare(outputMem, outputMem, inputMem, count, dataType,
                    algResResp_->slaveStreams[ringIndex], HcclReduceOp::HCCL_REDUCE_RESERVED, OUTER_BRIDGE_RANK_ID,
                    singleRingSliceZero, baseOffset, ringNics[ringIndex%halfRingSize]);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u],all gather(ring) prepare "\
                    "failed,return[%d]", ringIndex, ret), ret);
                ret = executor->RegisterProfiler(
                    ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                    (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + outerRingCommInfo.localRank,
                    profStage, HCCL_EXEC_STEP_NOT_SET, algResResp_->slaveStreams[ringIndex]);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u],all gather(ring) register "\
                    "Profiler failed,return[%d]", ringIndex, ret), ret);

                ret = RunTemplate(executor, outerRingCommInfo);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u],all gather(ring)"\
                    " run failed,return[%d]", ringIndex, ret), ret);

                ret = LocalNotify::Post(algResResp_->slaveStreams[ringIndex], dispatcher_,
                    algResResp_->notifiesM2S[ringIndex], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u] record failed",
                    ringIndex), ret);
            }

            ret = LocalNotify::Post(stream, dispatcher_, algResResp_->notifiesS2M[ringIndex], profStage);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u] record failed", ringIndex), ret);
        } else { // 主环
            std::unique_ptr<ExecutorBase> executor;
            if (opInfo != nullptr) {
                executor.reset(new (std::nothrow) AllGatherRingConcurrentDirect(
                    dispatcher_, opInfo, topoAttr_.userRank, subStreamsInOneRing, mainSignalsInOneRing,
                    subSignalsInOneRing, rankOrder, userMemOutputSlices, isSdma));
            } else {
                executor.reset(new (std::nothrow) AllGatherRing(dispatcher_));
            }
            CHK_SMART_PTR_NULL(executor);
            ret = executor->Prepare(outputMem, outputMem, inputMem, count, dataType, stream, HCCL_REDUCE_RESERVED,
                OUTER_BRIDGE_RANK_ID, singleRingSliceZero, baseOffset, ringNics[ringIndex%halfRingSize]);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u],all gather(ring) prepare"\
                " failed,return[%d]", ringIndex, ret), ret);

            ret = executor->RegisterProfiler(
                ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + outerRingCommInfo.localRank,
                profStage, HCCL_EXEC_STEP_NOT_SET, stream);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u],all gather(ring) register "\
                "Profiler failed, return[%d]", ringIndex, ret), ret);

            ret = RunTemplate(executor, outerRingCommInfo);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u],all gather(ring) run failed,"\
                "return[%d]", ringIndex, ret), ret);

            for (u32 ring = 0; ring < (ringNum - 1); ring++) {
                if (!topoMatcher_->GetExternalInputHcclEnableFfts() &&
                    workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                    algResResp_->threadManage[ring]->WaitDone(); // 单算子模式，等待线程处理完成信号
                }
                ret = LocalNotify::Wait(stream, dispatcher_, algResResp_->notifiesM2S[ring], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u] wait failed", ring), ret);
            }
        }
    }

    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}

HcclResult CollCommExecutor::MultiRingAsymAllGather(const std::string &tag, DeviceMem inputMem,
    DeviceMem outputMem, const u64 count, const HcclDataType dataType,
    const std::vector<std::pair<bool, std::vector<Slice>>> multRingsSliceZero,
    Stream stream, s32 profStage, const u64 baseOffset, const HcomCollOpInfo *opInfo,
    const std::vector<std::pair<bool, std::vector<Slice>>> multRingsUserMemSlice)
{
    HcclResult ret = HCCL_SUCCESS;
    u32 ringNum = multRingsSliceZero.size(); // 环数, 当前为4环

    std::vector<std::vector<u32>> ringNics;
    CHK_RET(GetRingNics(tag, ringNics));
    auto halfRingSize = ringNum;
    if (ringNum > RDMA_PLANE_NUM_IN_NPRING_DOUBLE) {
        halfRingSize = ringNum / 2; // 2环
    }
    // 拿到ring环映射关系
    SubCommInfo outerZeroCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    auto nicList = topoAttr_.nicList;
    std::vector<std::vector<u32>> multiRingsOrder =
        GetRingsOrderByTopoType(outerZeroCommInfo.localRankSize, topoType_, nicList);

    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    for (u32 ringIndex = 0; ringIndex < ringNum; ringIndex++) {
        std::vector<Slice> singleRingSliceZero = multRingsSliceZero[ringIndex].second; // 取出sdma/rdma的数据块
        CHK_PRT_RET(singleRingSliceZero.empty(), HCCL_ERROR("[CollCommExecutor][MultiRingAllGatherConcurrent]"\
            "singleRingSliceZero is empty"), HCCL_E_INTERNAL);

        // 910_93场景 生成userMemOut_上对应的slices
        std::vector<Slice> userMemOutputSlices;
        if (multRingsUserMemSlice.size() == 0) {
            CHK_RET(CalUserMemSlices(dataType, opInfo, singleRingSliceZero, ringIndex, multiRingsOrder,
                userMemOutputSlices));
        } else {
            userMemOutputSlices = multRingsUserMemSlice[ringIndex].second;
        }
        std::vector<u32> rankOrder;
        u32 commIndex = ringIndex % halfRingSize;
        CHK_RET(GetRankOrder(multiRingsOrder, commIndex, rankOrder));

        SubCommInfo outerRingCommInfo = multRingsSliceZero[ringIndex].first ?
            GetSubCommInfo(COMM_LEVEL0, commIndex) : GetSubCommInfo(COMM_LEVEL0_RDMA, commIndex);

        u32 rankSize = outerRingCommInfo.localRankSize;
        u32 ringIndexOp = ringIndex;

        // 910_93场景 准备环中的从流
        std::vector<Stream>       subStreamsInOneRing;
        std::vector<std::shared_ptr<LocalNotify>> mainSignalsInOneRing;
        std::vector<std::shared_ptr<LocalNotify>> subSignalsInOneRing;
        if (opInfo != nullptr) {
            CHK_RET(GetSubStreamInfoOnOneRing(ringIndex, subStreamsInOneRing, mainSignalsInOneRing,
                                              subSignalsInOneRing));
        }
        bool isSdma = multRingsSliceZero[ringIndex].first;
        if (ringIndex != (ringNum)) { // 最后一个环是主stream，所以这里减1，符合条件的走从stream
            if (!topoMatcher_->GetExternalInputHcclEnableFfts() &&
                workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                if (opInfo != nullptr) {
                    ExecutorType type = isSdma ?
                        ExecutorType::ALLGATHER_HALF_RING_DIRECT : ExecutorType::ALLGATHER_HALF_RING_DIRECT_RDMA;
                    algResResp_->threadManage[ringIndex]->Prepare(
                        outputMem, outputMem, inputMem, count, dataType,
                        algResResp_->slaveStreams[ringIndex], HcclReduceOp::HCCL_REDUCE_RESERVED, OUTER_BRIDGE_RANK_ID,
                        singleRingSliceZero, baseOffset, ringNics[ringIndex%halfRingSize], tag, profStage,
                        outerRingCommInfo, algResResp_->notifiesS2M[ringIndex], algResResp_->notifiesM2S[ringIndex],
                        ringIndex, type, 0, opInfo, subStreamsInOneRing,
                        mainSignalsInOneRing, subSignalsInOneRing, rankOrder, userMemOutputSlices);
                } else {
                    algResResp_->threadManage[ringIndex]->Prepare(outputMem, outputMem, inputMem, count, dataType,
                        algResResp_->slaveStreams[ringIndex], HcclReduceOp::HCCL_REDUCE_RESERVED, OUTER_BRIDGE_RANK_ID,
                        singleRingSliceZero, baseOffset, ringNics[ringIndex%halfRingSize], tag, profStage,
                        outerRingCommInfo, algResResp_->notifiesS2M[ringIndex], algResResp_->notifiesM2S[ringIndex],
                        ringIndex, ExecutorType::ALLGATHER_HALF_RING);
                }
                algResResp_->threadManage[ringIndex]->NotifyStart();    // 给线程发信号启动处理
            } else {
                ret = LocalNotify::Wait(algResResp_->slaveStreams[ringIndex], dispatcher_,
                    algResResp_->notifiesS2M[ringIndex], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u] wait failed",
                    ringIndex), ret);
                // 如何判断是否环内是否有数据, 以ring的第一个rank的 size为判断依据
                std::unique_ptr<ExecutorBase> executor;
                if (opInfo != nullptr) {
                    executor.reset(new (std::nothrow) AllGatherHalfRingDirect(
                        dispatcher_, opInfo, topoAttr_.userRank, subStreamsInOneRing, mainSignalsInOneRing,
                        subSignalsInOneRing, rankOrder, commIndex, userMemOutputSlices, isSdma));
                } else {
                    executor.reset(new (std::nothrow) AllGatherHalfRing(dispatcher_, commIndex));
                }
                CHK_SMART_PTR_NULL(executor);

                std::cout << "ready to prepre in sub ring" << '\n';
                ret = executor->Prepare(outputMem, outputMem, inputMem, count, dataType,
                    algResResp_->slaveStreams[ringIndex], HcclReduceOp::HCCL_REDUCE_RESERVED, OUTER_BRIDGE_RANK_ID,
                    singleRingSliceZero, baseOffset, ringNics[ringIndex%halfRingSize]);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u],all gather(ring) prepare "\
                    "failed,return[%d]", ringIndex, ret), ret);
                ret = executor->RegisterProfiler(
                    ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                    (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + outerRingCommInfo.localRank,
                    profStage, HCCL_EXEC_STEP_NOT_SET, algResResp_->slaveStreams[ringIndex]);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u],all gather(ring) register "\
                    "Profiler failed,return[%d]", ringIndex, ret), ret);

                ret = RunTemplate(executor, outerRingCommInfo);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u],all gather(ring)"\
                    " run failed,return[%d]", ringIndex, ret), ret);

                ret = LocalNotify::Post(algResResp_->slaveStreams[ringIndex], dispatcher_,
                    algResResp_->notifiesM2S[ringIndex], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u] record failed",
                    ringIndex), ret);
            }

            ret = LocalNotify::Post(stream, dispatcher_, algResResp_->notifiesS2M[ringIndex], profStage);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u] record failed", ringIndex), ret);
        } else { // 主环
            std::unique_ptr<ExecutorBase> executor;
            if (opInfo != nullptr) {
                executor.reset(new (std::nothrow) AllGatherHalfRingDirect(
                    dispatcher_, opInfo, topoAttr_.userRank, subStreamsInOneRing, mainSignalsInOneRing,
                    subSignalsInOneRing, rankOrder, commIndex, userMemOutputSlices, isSdma));
            } else {
                executor.reset(new (std::nothrow) AllGatherHalfRing(dispatcher_, commIndex));
            }
            CHK_SMART_PTR_NULL(executor);

            std::cout << "ready to prepre in main ring" << '\n';
            ret = executor->Prepare(outputMem, outputMem, inputMem, count, dataType, stream, HCCL_REDUCE_RESERVED,
                OUTER_BRIDGE_RANK_ID, singleRingSliceZero, baseOffset, ringNics[ringIndex%halfRingSize]);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u],all gather(ring) prepare"\
                " failed,return[%d]", ringIndex, ret), ret);

            ret = executor->RegisterProfiler(
                ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + outerRingCommInfo.localRank,
                profStage, HCCL_EXEC_STEP_NOT_SET, stream);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u],all gather(ring) register "\
                "Profiler failed, return[%d]", ringIndex, ret), ret);

            ret = RunTemplate(executor, outerRingCommInfo);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u],all gather(ring) run failed,"\
                "return[%d]", ringIndex, ret), ret);

            for (u32 ring = 0; ring < (ringNum - 1); ring++) {
                if (!topoMatcher_->GetExternalInputHcclEnableFfts() &&
                    workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                    algResResp_->threadManage[ring]->WaitDone(); // 单算子模式，等待线程处理完成信号
                }
                ret = LocalNotify::Wait(stream, dispatcher_, algResResp_->notifiesM2S[ring], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u] wait failed", ring), ret);
            }
        }
    }

    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}

HcclResult CollCommExecutor::Level1AllGatherConcurrent(DeviceMem inputMem, DeviceMem outputMem,const u64 count,
    const HcclDataType dataType, Stream stream, s32 profStage,std::vector<Slice> &level1DataSegsSlice, u32 syncTrans)
{
    std::vector<std::pair<bool, std::vector<Slice>>> innerMultSlice;
    std::vector<Slice> level1DataSegsSliceSdma;
    std::vector<Slice> level1DataSegsSliceRdma;
    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 level0ServerIndex = outerCommInfo.localRank;
    SubCommInfo innerCommInfo = GetSubCommInfo(COMM_LEVEL1, level0ServerIndex);
    CHK_RET(CheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1));
    SubCommInfo level2CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);
    HcclResult ret = HCCL_SUCCESS;
    innerMultSlice.resize(RDMA_PLANE_NUM_IN_NPRING_DOUBLE);

    for (u32 i = 0; i < innerCommInfo.localRankSize; i++) {
        Slice sdmaSlice;
        Slice rdmaSlice;
        u64 sdmaSliceSize =
            ((level1DataSegsSlice[i].size <= HCCL_MIN_SLICE_ALIGN_910_93) || (syncTrans == MAX_SPLIT_VALUE))
                ? level1DataSegsSlice[i].size
                : ((syncTrans * level1DataSegsSlice[i].size / MAX_SPLIT_VALUE) / HCCL_MIN_SLICE_ALIGN_910_93) *
                      HCCL_MIN_SLICE_ALIGN_910_93;
        sdmaSlice.size = sdmaSliceSize;
        sdmaSlice.offset = level1DataSegsSlice[i].offset;
        rdmaSlice.size = level1DataSegsSlice[i].size - sdmaSliceSize;
        rdmaSlice.offset = level1DataSegsSlice[i].offset + sdmaSliceSize;
        level1DataSegsSliceSdma.push_back(sdmaSlice);
        level1DataSegsSliceRdma.push_back(rdmaSlice);
        HCCL_DEBUG("Level1 index:[%u], Orignal [offset %llu, size %llu], sdma [offset %llu, size %llu], "
                   "rdma [offset %llu, size %llu]", i, level1DataSegsSlice[i].offset, level1DataSegsSlice[i].size,
            sdmaSlice.offset, sdmaSlice.size, rdmaSlice.offset, rdmaSlice.size);
    }
    innerMultSlice[0] = std::make_pair(true, level1DataSegsSliceSdma);
    innerMultSlice[1] = std::make_pair(false, level1DataSegsSliceRdma);

    u32 commPlaneNum = innerMultSlice.size();
    for (u32 planeIndex = 0; planeIndex < commPlaneNum; planeIndex++) {
        std::vector<Slice> &singleSlice = innerMultSlice[planeIndex].second;
        SubCommInfo innerRdmaCommInfo = GetSubCommInfo(COMM_LEVEL1_RDMA, level0ServerIndex);
        SubCommInfo level1CommInfo = innerMultSlice[planeIndex].first ? innerCommInfo : innerRdmaCommInfo;
        std::unique_ptr<ExecutorBase> innerExecutor;
        if (UseInterServerNBAlgo(algType_)) {
            innerExecutor.reset(new (std::nothrow) AllGatherNB(dispatcher_));
            HCCL_INFO("allgather ring: using nonuniform-bruck algo inter-server.");
        } else {
            innerExecutor.reset(new (std::nothrow) AllGatherRing(dispatcher_));
            HCCL_INFO("allgather ring: using ring algo inter-server.");
        }
        CHK_SMART_PTR_NULL(innerExecutor);

        if (planeIndex != (commPlaneNum - 1)) {
            ret = LocalNotify::Wait(
                algResResp_->slaveStreams[planeIndex], dispatcher_, algResResp_->notifiesS2M[planeIndex], profStage);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("stream[%u] wait failed", planeIndex), ret);

            CHK_RET(innerExecutor->Prepare(outputMem, outputMem, inputMem, count,
                dataType, algResResp_->slaveStreams[planeIndex], HCCL_REDUCE_RESERVED,
                INVALID_VALUE_RANKID, singleSlice, 0));

            CHK_RET(innerExecutor->RegisterProfiler(
                (level1CommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2CommInfo.localRank,
                profStage, HCCL_EXEC_STEP_NOT_SET, algResResp_->slaveStreams[planeIndex]));

            CHK_RET(RunTemplate(innerExecutor, level1CommInfo));
            ret = LocalNotify::Post(
                algResResp_->slaveStreams[planeIndex], dispatcher_, algResResp_->notifiesM2S[planeIndex], profStage);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS, HCCL_ERROR("[collAllGather]level1 stream[%u] record failed", planeIndex), ret);
            // 主环record启动从环
            ret = LocalNotify::Post(stream, dispatcher_, algResResp_->notifiesS2M[planeIndex], profStage);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS, HCCL_ERROR("[collAllGather]level1 stream[%u] record failed", planeIndex), ret);
        } else {
            CHK_RET(innerExecutor->Prepare(outputMem, outputMem, inputMem, count, dataType, stream,
                HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, singleSlice, 0));
            CHK_RET(innerExecutor->RegisterProfiler(
                (level1CommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2CommInfo.localRank,
                profStage, HCCL_EXEC_STEP_NOT_SET, stream));

            CHK_RET(RunTemplate(innerExecutor, level1CommInfo));
            for (u32 ring = 0; ring < (commPlaneNum - 1); ring++) {
                ret = LocalNotify::Wait(stream, dispatcher_, algResResp_->notifiesM2S[ring], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("stream[%u] wait failed", ring), ret);
            }
        }
    }
    HCCL_INFO("Level1AllGatherConcurrent run success");
    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}

HcclResult CollCommExecutor::CollectMultiRingsUserMemSlices(u32 ringNum, const HcclDataType dataType,
    const HcomCollOpInfo *opInfo, const std::vector<std::vector<Slice>> &multRingsSliceZero,
    const std::vector<std::vector<u32>> &multiRingsOrder,
    const std::vector<std::vector<Slice>> &multRingsUserMemSlice,
    std::vector<std::vector<Slice>> &userMemSlicesOfMultiRings)
{
    for (u32 ringIndex = 0; ringIndex < ringNum; ringIndex++) {
        std::vector<Slice> singleRingSliceZero = multRingsSliceZero[ringIndex];
        CHK_PRT_RET(singleRingSliceZero.empty(),
            HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatter]singleRingSliceZero is empty"), HCCL_E_INTERNAL);
        std::vector<Slice> userMemSlices;
        if (multRingsUserMemSlice.size() == 0) {
            CHK_RET(CalUserMemSlices(dataType, opInfo, singleRingSliceZero, ringIndex, multiRingsOrder,
                userMemSlices));
        } else {
            userMemSlices = multRingsUserMemSlice[ringIndex];
        }
        userMemSlicesOfMultiRings.push_back(userMemSlices);
    }
    return HCCL_SUCCESS;
}

HcclResult CollCommExecutor::CollectMultiRingsRankOrder(u32 ringNum,
    const std::vector<std::vector<u32>> &multiRingsOrder,
    std::vector<std::vector<u32>> &rankOrders)
{
    for (u32 ringIndex = 0; ringIndex < ringNum; ringIndex++) {
        std::vector<u32> rankOrder;
        CHK_RET(GetRankOrder(multiRingsOrder, ringIndex, rankOrder));
        rankOrders.push_back(rankOrder);
    }
    return HCCL_SUCCESS;
}

HcclResult CollCommExecutor::MultiRingReduceScatter(const std::string &tag, DeviceMem inputMem, DeviceMem outputMem,
    const u64 count, const HcclDataType dataType, const HcclReduceOp reductionOp,
    const std::vector<std::vector<Slice> > multRingsSliceZero, Stream stream, s32 profStage,
    const u64 baseOffset, const HcomCollOpInfo *opInfo,
    const std::vector<std::vector<Slice>> multRingsUserMemSlice)
{
    HCCL_INFO("[MultiRingReduceScatter] MultiRingReduceScatter starts");
    HcclResult ret = HCCL_SUCCESS;
    u32 ringNum = multRingsSliceZero.size();
    CHK_RET(CheckCommSize(COMM_LEVEL0, ringNum));

    std::vector<std::vector<u32>> ringNics;
    CHK_RET(GetRingNics(tag, ringNics));
    // 拿到ring环映射关系
    SubCommInfo outerZeroCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    auto nicList = topoAttr_.nicList;
    std::vector<std::vector<u32>> multiRingsOrder =
        GetRingsOrderByTopoType(outerZeroCommInfo.localRankSize, topoType_, nicList);

    u64 reduceAttr = GetReduceAttr(inputMem, outputMem, dataType, reductionOp);

    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    for (u32 ringIndex = 0; ringIndex < ringNum; ringIndex++) {
        std::vector<Slice> singleRingSliceZero = multRingsSliceZero[ringIndex];
        CHK_PRT_RET(singleRingSliceZero.empty(),
            HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatter]singleRingSliceZero is empty"), HCCL_E_INTERNAL);

        // 生成userMemIn_上对应的slices
        std::vector<Slice> userMemInputSlices;
        if (multRingsUserMemSlice.size() == 0) {
            CHK_RET(CalUserMemSlices(dataType, opInfo, singleRingSliceZero, ringIndex, multiRingsOrder,
                userMemInputSlices));
        } else {
            userMemInputSlices = multRingsUserMemSlice[ringIndex];
        }

        std::vector<u32> rankOrder;
        CHK_RET(GetRankOrder(multiRingsOrder, ringIndex, rankOrder));

        SubCommInfo outerRingCommInfo = GetSubCommInfo(COMM_LEVEL0, ringIndex);
        u32 rankSize = outerRingCommInfo.localRankSize;
        u32 ringIndexOp = ringIndex;

        std::vector<Stream>       subStreamsInOneRing;
        std::vector<std::shared_ptr<LocalNotify>> mainSignalsInOneRing;
        std::vector<std::shared_ptr<LocalNotify>> subSignalsInOneRing;
        if (opInfo != nullptr) {
            CHK_RET(GetSubStreamInfoOnOneRing(ringIndex, subStreamsInOneRing, mainSignalsInOneRing,
                                              subSignalsInOneRing));
        }
        if (ringIndex != (ringNum - 1)) {  // 0~ringNum-2的环
            if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) { // offline
                ret = StreamActiveManager::GetInstance(topoAttr_.deviceLogicId).StreamActive(
                    algResResp_->slaveStreams[ringIndex].ptr(), stream.ptr());
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatter]active stream[%u], failed",
                    ringIndex), ret);
            }
            if (!topoMatcher_->GetExternalInputHcclEnableFfts() &&
                workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                /* 更新线程参数 */
                if (opInfo != nullptr) {
                    algResResp_->threadManage[ringIndex]->Prepare(
                        inputMem, inputMem, outputMem, count, dataType, algResResp_->slaveStreams[ringIndex], reductionOp,
                        OUTER_BRIDGE_RANK_ID, singleRingSliceZero, baseOffset, ringNics[ringIndex], tag, profStage,
                        outerRingCommInfo, algResResp_->notifiesS2M[ringIndex], algResResp_->notifiesM2S[ringIndex],
                        ringIndex, ExecutorType::REDUCE_SCATTER_RING_DIRECT, reduceAttr, opInfo,
                        subStreamsInOneRing, mainSignalsInOneRing, subSignalsInOneRing, rankOrder,
                        userMemInputSlices);
                } else {
                    algResResp_->threadManage[ringIndex]->Prepare(inputMem, inputMem, outputMem, count, dataType,
                        algResResp_->slaveStreams[ringIndex], reductionOp, OUTER_BRIDGE_RANK_ID, singleRingSliceZero,
                        baseOffset, ringNics[ringIndex], tag, profStage, outerRingCommInfo,
                        algResResp_->notifiesS2M[ringIndex], algResResp_->notifiesM2S[ringIndex], ringIndex,
                        ExecutorType::REDUCE_SCATTER_RING, reduceAttr);
                }

                algResResp_->threadManage[ringIndex]->NotifyStart(); // 给线程发通知启动线程执行
            } else {
                std::unique_ptr<ExecutorBase> executor;
                if (opInfo != nullptr) {
                    executor.reset(new (std::nothrow) ReduceScatterRingConcurrentDirect(
                        dispatcher_, reduceAttr, opInfo, topoAttr_.userRank, subStreamsInOneRing,
                        mainSignalsInOneRing, subSignalsInOneRing, rankOrder, userMemInputSlices));
                } else {
                    executor.reset(new (std::nothrow) ReduceScatterRing(dispatcher_, reduceAttr));
                }
                CHK_SMART_PTR_NULL(executor);

                ret = LocalNotify::Wait(algResResp_->slaveStreams[ringIndex], dispatcher_,
                    algResResp_->notifiesS2M[ringIndex], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatter]stream[%u] wait failed", ringIndex), ret);
                ret = executor->Prepare(inputMem, inputMem, outputMem, count, dataType,
                    algResResp_->slaveStreams[ringIndex], reductionOp, OUTER_BRIDGE_RANK_ID,
                    singleRingSliceZero, baseOffset, ringNics[ringIndex]);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatter]stream[%u],reduce scatter(ring) "\
                    "prepare failed,return[%d]", ringIndex, ret), ret);
                ret = executor->RegisterProfiler(
                    ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                    (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + outerRingCommInfo.localRank,
                    profStage, HCCL_EXEC_STEP_NOT_SET, algResResp_->slaveStreams[ringIndex]);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatter]stream[%u],reduce scatter(ring) "\
                    "register Profiler failed,return[%d]", ringIndex, ret), ret);

                ret = RunTemplate(executor, outerRingCommInfo);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatter]stream[%u],reduce scatter(ring) run "\
                    "failed,return[%d]", ringIndex, ret), ret);

                ret = LocalNotify::Post(algResResp_->slaveStreams[ringIndex], dispatcher_,
                    algResResp_->notifiesM2S[ringIndex], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatter]stream[%u] record failed", ringIndex), ret);
            }
            /* 主环record启动从环 */
            ret = LocalNotify::Post(stream, dispatcher_, algResResp_->notifiesS2M[ringIndex], profStage);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatter]stream[%u] record failed", ringIndex), ret);
        } else { // 主环 最后一个环
            std::unique_ptr<ExecutorBase> executor;
            if (opInfo != nullptr) {
                executor.reset(new (std::nothrow) ReduceScatterRingConcurrentDirect(
                    dispatcher_, reduceAttr, opInfo, topoAttr_.userRank, subStreamsInOneRing, mainSignalsInOneRing,
                    subSignalsInOneRing, rankOrder, userMemInputSlices));
            } else {
                executor.reset(new (std::nothrow) ReduceScatterRing(dispatcher_, reduceAttr));
            }
            CHK_SMART_PTR_NULL(executor);
            ret = executor->Prepare(inputMem, inputMem, outputMem, count, dataType, stream,
                reductionOp, OUTER_BRIDGE_RANK_ID, singleRingSliceZero, baseOffset, ringNics[ringIndex]);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatter]stream[%u],reduce scatter(ring) prepare "\
                "failed,return[%d]", ringIndex, ret), ret);

            ret = executor->RegisterProfiler(
                ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + outerRingCommInfo.localRank,
                profStage, HCCL_EXEC_STEP_NOT_SET, stream);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatter]stream[%u],reduce scatter(ring) register "\
                "Profiler failed,return[%d]", ringIndex, ret), ret);

            ret = RunTemplate(executor, outerRingCommInfo);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatter]stream[%u],reduce scatter(ring) run "\
                "failed,return[%d]", ringIndex, ret), ret);
            for (u32 ring = 0; ring < (ringNum - 1); ring++) {
                if (!topoMatcher_->GetExternalInputHcclEnableFfts() &&
                    workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                    algResResp_->threadManage[ring]->WaitDone();
                }
                /* 等待executor执行完毕 */
                ret = LocalNotify::Wait(stream, dispatcher_, algResResp_->notifiesM2S[ring], profStage);

                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatter]stream[%u] wait failed", ring), ret);
            }
        }
    }

    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}

HcclResult CollCommExecutor::MultiRingGather(const std::string &tag, DeviceMem inputMem, DeviceMem outputMem,
    const u64 count, const HcclDataType dataType, const std::vector<std::vector<Slice> > multRingsSliceZero,
    HcclReduceOp op, u32 root, Stream stream, s32 profStage)
{
    u32 ringNum = multRingsSliceZero.size();
    std::vector<std::vector<u32>> ringNics;
    CHK_RET(GetRingNics(tag, ringNics));

    HcclResult ret;

    for (u32 ringIndex = 0; ringIndex < ringNum; ringIndex++) {
        std::vector<Slice> singleRingSliceZero = multRingsSliceZero[ringIndex];
        CHK_PRT_RET(singleRingSliceZero.empty(),
            HCCL_ERROR("[CommonOperator][MultiRingGather]singleRingSliceZero is empty"), HCCL_E_INTERNAL);

        SubCommInfo outerRingCommInfo = GetSubCommInfo(COMM_LEVEL0, ringIndex);
        u32 rankSize = outerRingCommInfo.localRankSize;
        u32 rootRank = 0;
        ret = GetRankByUserRank(COMM_LEVEL0, ringIndex, root, rootRank);
        CHK_PRT_RET(ret == HCCL_E_PARA,
            HCCL_ERROR("[CommonOperator][MultiRingGather]invalid root rank[%u] to get user rank", root), ret);

        std::unique_ptr<ExecutorBase> executor = nullptr;
        EXECEPTION_CATCH((executor = std::make_unique<GatherRing>(dispatcher_)), return HCCL_E_PTR);

        if (ringIndex != (ringNum - 1)) {  // 0~ringNum-2的环
            if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) { // offline
                CHK_RET(StreamActiveManager::GetInstance(topoAttr_.deviceLogicId).StreamActive(
                    algResResp_->slaveStreams[ringIndex].ptr(), stream.ptr()));
            }
            ret = LocalNotify::Wait(algResResp_->slaveStreams[ringIndex], dispatcher_,
                algResResp_->notifiesS2M[ringIndex], profStage);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[CommonOperator][MultiRingGather]in stream[%u] wait failed", \
                ringIndex), ret);
            if (singleRingSliceZero[0].size != 0) {
            ret = executor->Prepare(inputMem, outputMem, outputMem, count, dataType,
                                    algResResp_->slaveStreams[ringIndex], op, rootRank, singleRingSliceZero, 0,
                                    ringNics[ringIndex]);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CommonOperator][MultiRingGather]stream[%u],gather(ring) prepare failed, "\
                "return[%d]", ringIndex, ret), ret);

            ret = executor->RegisterProfiler(outerRingCommInfo.localRank, profStage, HCCL_EXEC_STEP_NOT_SET,
                algResResp_->slaveStreams[ringIndex]);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CommonOperator][MultiRingGather]stream[%u], gather(ring) register profiler "\
                "failed,return[%d]", ringIndex, ret), ret);

            ret = RunTemplate(executor, outerRingCommInfo);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CommonOperator][MultiRingGather]stream[%u],gather(ring) run failed,return[%d]",
                ringIndex, ret), ret);
            }
            ret = LocalNotify::Post(algResResp_->slaveStreams[ringIndex], dispatcher_, algResResp_->notifiesM2S[ringIndex],
                profStage);

            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[CommonOperator][MultiRingGather]stream[%u] record failed", \
                ringIndex), ret);

            ret = LocalNotify::Post(stream, dispatcher_, algResResp_->notifiesS2M[ringIndex], profStage);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[CommonOperator][MultiRingGather]stream[%u] record failed", \
                ringIndex), ret);
        } else {  // 主环
            executor.reset(new (std::nothrow) GatherRing(dispatcher_));
            CHK_SMART_PTR_NULL(executor);

            ret = executor->Prepare(inputMem, outputMem, outputMem, count, dataType, stream,
                op, rootRank, singleRingSliceZero, 0, ringNics[ringIndex]);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CommonOperator][MultiRingGather]stream[%u],gather(ring) prepare failed, "\
                "return[%d]", ringIndex, ret), ret);

            ret = executor->RegisterProfiler(((ringIndex + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + outerRingCommInfo.localRank,
                profStage, HCCL_EXEC_STEP_NOT_SET, stream);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CommonOperator][MultiRingGather]stream[%u], gather(ring) register "\
                "profiler failed,return[%d]", ringIndex, ret), ret);

            ret = RunTemplate(executor, outerRingCommInfo);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CommonOperator][MultiRingGather]stream[%u],gather(ring) run failed, "\
                "return[%d]", ringIndex, ret), ret);
            for (u32 ring = 0; ring < (ringNum - 1); ring++) {
                /* 等待executor执行完毕 , 当前环没有分配数据，跳过此环处理，继续下一个环 */
                ret = LocalNotify::Wait(stream, dispatcher_, algResResp_->notifiesM2S[ring], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CommonOperator][MultiRingGather]stream[%u] wait failed", ring), ret);
            }
        }
    }

    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}

HcclResult CollCommExecutor::MultiRingReduceScatterConcurrent(const std::string &tag, DeviceMem inputMem,
    DeviceMem outputMem, const u64 count, const HcclDataType dataType, const HcclReduceOp reductionOp,
    const std::vector<std::pair<bool, std::vector<Slice>>> multRingsSliceZero, Stream stream, s32 profStage,
    const u64 baseOffset, const HcomCollOpInfo *opInfo,
    const std::vector<std::pair<bool, std::vector<Slice>>> multRingsUserMemSlice)
{
    HcclResult ret = HCCL_SUCCESS;
    u32 ringNum = multRingsSliceZero.size();

    std::vector<std::vector<u32>> ringNics;
    CHK_RET(GetRingNics(tag, ringNics));
    u32 halfRingSize = ringNum;
    u32 DoubleRing = 2;
    if (ringNum > RDMA_PLANE_NUM_IN_NPRING_DOUBLE) {
        halfRingSize = ringNum / DoubleRing;
    }

    // 拿到ring环映射关系
    SubCommInfo outerZeroCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    auto nicList = topoAttr_.nicList;
    std::vector<std::vector<u32>> multiRingsOrder =
        GetRingsOrderByTopoType(outerZeroCommInfo.localRankSize, topoType_, nicList);

    u64 reduceAttr = GetReduceAttr(inputMem, outputMem, dataType, reductionOp);

    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    for (u32 ringIndex = 0; ringIndex < ringNum; ringIndex++) {
        std::vector<Slice> singleRingSliceZero = multRingsSliceZero[ringIndex].second;
        CHK_PRT_RET(singleRingSliceZero.empty(),
            HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatterConcurrent]singleRingSliceZero is empty"),
            HCCL_E_INTERNAL);

        // 生成userMemIn_上对应的slices
        std::vector<Slice> userMemInputSlices;
        u32 commIndex = ringIndex % halfRingSize;
        if (multRingsUserMemSlice.size() == 0) {
            CHK_RET(CalUserMemSlices(dataType, opInfo, singleRingSliceZero, ringIndex, multiRingsOrder,
                userMemInputSlices));
        } else {
            userMemInputSlices = multRingsUserMemSlice[ringIndex].second;
        }
        std::vector<u32> rankOrder;
        CHK_RET(GetRankOrder(multiRingsOrder, commIndex, rankOrder));

        SubCommInfo outerRingCommInfo = multRingsSliceZero[ringIndex].first ?
            GetSubCommInfo(COMM_LEVEL0, commIndex) : GetSubCommInfo(COMM_LEVEL0_RDMA, commIndex);
        u32 rankSize = outerRingCommInfo.localRankSize;
        u32 ringIndexOp = ringIndex;

        std::vector<Stream>       subStreamsInOneRing;
        std::vector<std::shared_ptr<LocalNotify>> mainSignalsInOneRing;
        std::vector<std::shared_ptr<LocalNotify>> subSignalsInOneRing;
        if (opInfo != nullptr) {
            CHK_RET(GetSubStreamInfoOnOneRing(ringIndex, subStreamsInOneRing, mainSignalsInOneRing,
                                              subSignalsInOneRing));
        }
        bool isSdma = multRingsSliceZero[ringIndex].first;
        if (ringIndex != (ringNum - 1)) {  // 0~ringNum-2的环
            if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) { // offline
                ret = StreamActiveManager::GetInstance(topoAttr_.deviceLogicId).StreamActive(
                    algResResp_->slaveStreams[ringIndex].ptr(), stream.ptr());
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatterConcurrent]active stream[%u], failed",
                        ringIndex), ret);
            }

            if (!topoMatcher_->GetExternalInputHcclEnableFfts() &&
                workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                /* 更新线程参数 */
                if (opInfo != nullptr) {
                    ExecutorType type = isSdma ?
                       ExecutorType::REDUCE_SCATTER_RING_DIRECT : ExecutorType::REDUCE_SCATTER_RING_DIRECT_RDMA;
                    algResResp_->threadManage[ringIndex]->Prepare(
                        inputMem, inputMem, outputMem, count, dataType, algResResp_->slaveStreams[ringIndex], reductionOp,
                        OUTER_BRIDGE_RANK_ID, singleRingSliceZero, baseOffset, ringNics[ringIndex % halfRingSize], tag,
                        profStage, outerRingCommInfo, algResResp_->notifiesS2M[ringIndex],
                        algResResp_->notifiesM2S[ringIndex], ringIndex, type,
                        reduceAttr, opInfo, subStreamsInOneRing, mainSignalsInOneRing, subSignalsInOneRing, rankOrder,
                        userMemInputSlices);
                } else {
                    algResResp_->threadManage[ringIndex]->Prepare(inputMem, inputMem, outputMem, count, dataType,
                        algResResp_->slaveStreams[ringIndex], reductionOp, OUTER_BRIDGE_RANK_ID, singleRingSliceZero,
                        baseOffset, ringNics[ringIndex % halfRingSize], tag, profStage, outerRingCommInfo,
                        algResResp_->notifiesS2M[ringIndex], algResResp_->notifiesM2S[ringIndex], ringIndex,
                        ExecutorType::REDUCE_SCATTER_RING, reduceAttr);
                }

                algResResp_->threadManage[ringIndex]->NotifyStart(); // 给线程发通知启动线程执行
            } else {
                std::unique_ptr<ExecutorBase> executor;
                if (opInfo != nullptr) {
                    executor.reset(new (std::nothrow) ReduceScatterRingConcurrentDirect(
                        dispatcher_, reduceAttr, opInfo, topoAttr_.userRank, subStreamsInOneRing,
                        mainSignalsInOneRing, subSignalsInOneRing, rankOrder, userMemInputSlices, isSdma));
                } else {
                    executor.reset(new (std::nothrow) ReduceScatterRing(dispatcher_, reduceAttr));
                }
                CHK_SMART_PTR_NULL(executor);

                ret = LocalNotify::Wait(algResResp_->slaveStreams[ringIndex], dispatcher_,
                    algResResp_->notifiesS2M[ringIndex], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatterConcurrent]stream[%u] wait failed", ringIndex),
                    ret);
                ret = executor->Prepare(inputMem, inputMem, outputMem, count, dataType,
                    algResResp_->slaveStreams[ringIndex], reductionOp, OUTER_BRIDGE_RANK_ID,
                    singleRingSliceZero, baseOffset, ringNics[ringIndex % halfRingSize]);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatterConcurrent]stream[%u],reduce scatter(ring) "\
                    "prepare failed,return[%d]", ringIndex, ret), ret);
                ret = executor->RegisterProfiler(
                    ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                    (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + outerRingCommInfo.localRank,
                    profStage, HCCL_EXEC_STEP_NOT_SET, algResResp_->slaveStreams[ringIndex]);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatterConcurrent]stream[%u],reduce scatter(ring) "\
                    "register Profiler failed,return[%d]", ringIndex, ret), ret);

                ret = RunTemplate(executor, outerRingCommInfo);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatterConcurrent]stream[%u],reduce scatter(ring)"\
                    " run failed,return[%d]", ringIndex, ret), ret);

                ret = LocalNotify::Post(algResResp_->slaveStreams[ringIndex], dispatcher_,
                    algResResp_->notifiesM2S[ringIndex], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatterConcurrent]stream[%u] record failed",
                    ringIndex),
                    ret);
            }
            /* 主环record启动从环 */
            ret = LocalNotify::Post(stream, dispatcher_, algResResp_->notifiesS2M[ringIndex], profStage);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatterConcurrent]stream[%u] record failed", ringIndex),
                ret);
        } else { // 主环 最后一个环
            std::unique_ptr<ExecutorBase> executor;
            if (opInfo != nullptr) {
                executor.reset(new (std::nothrow) ReduceScatterRingConcurrentDirect(
                    dispatcher_, reduceAttr, opInfo, topoAttr_.userRank, subStreamsInOneRing, mainSignalsInOneRing,
                    subSignalsInOneRing, rankOrder, userMemInputSlices, isSdma));
            } else {
                executor.reset(new (std::nothrow) ReduceScatterRing(dispatcher_, reduceAttr));
            }
            CHK_SMART_PTR_NULL(executor);
            ret = executor->Prepare(inputMem, inputMem, outputMem, count, dataType, stream,
                reductionOp, OUTER_BRIDGE_RANK_ID, singleRingSliceZero, baseOffset, ringNics[ringIndex % halfRingSize]);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatterConcurrent]stream[%u],reduce scatter(ring) "\
                " prepare failed,return[%d]", ringIndex, ret), ret);

            ret = executor->RegisterProfiler(
                ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + outerRingCommInfo.localRank,
                profStage, HCCL_EXEC_STEP_NOT_SET, stream);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatterConcurrent]stream[%u],reduce scatter(ring) "\
                "register Profiler failed,return[%d]", ringIndex, ret), ret);

            ret = RunTemplate(executor, outerRingCommInfo);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatterConcurrent]stream[%u],reduce scatter(ring) run "\
                "failed,return[%d]", ringIndex, ret), ret);
            for (u32 ring = 0; ring < (ringNum - 1); ring++) {
                if (!topoMatcher_->GetExternalInputHcclEnableFfts() &&
                    workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                    algResResp_->threadManage[ring]->WaitDone();
                }
                /* 等待executor执行完毕 */
                ret = LocalNotify::Wait(stream, dispatcher_, algResResp_->notifiesM2S[ring], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatterConcurrent]stream[%u] wait failed",
                    ring), ret);
            }
        }
    }

    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}

HcclResult CollCommExecutor::Level1ReduceScatterConcurrent(DeviceMem inputMem, DeviceMem scratchMem,const u64 count,
    const HcclDataType dataType, const HcclReduceOp reductionOp, Stream stream, s32 profStage,
    std::vector<Slice> &level1DataSegsSlice, u32 syncTrans, u64 reduceAttr)
{
    (void)profStage;
    std::vector<std::pair<bool, std::vector<Slice>>> innerMultSlice;
    innerMultSlice.resize(RDMA_PLANE_NUM_IN_NPRING_DOUBLE);
    std::vector<Slice> sdmaSlice;
    std::vector<Slice> rdmaSlice;
    for (u32 segsIndex = 0; segsIndex < level1DataSegsSlice.size(); segsIndex++) {
        u64 totalSize = level1DataSegsSlice[segsIndex].size;
        u64 sdmaSliceOffset = level1DataSegsSlice[segsIndex].offset;
        u64 sdmaSliceSize = ((totalSize <= HCCL_MIN_SLICE_ALIGN_910_93) || (syncTrans == MAX_SPLIT_VALUE)) ? totalSize
                                 : ((syncTrans * totalSize / MAX_SPLIT_VALUE) / HCCL_MIN_SLICE_ALIGN_910_93) *
                                       HCCL_MIN_SLICE_ALIGN_910_93;
        Slice sdmaSliceTmp;
        sdmaSliceTmp.offset = sdmaSliceOffset;
        sdmaSliceTmp.size = sdmaSliceSize;
        Slice rdmaSliceTmp;
        rdmaSliceTmp.offset = sdmaSliceOffset + sdmaSliceSize;
        rdmaSliceTmp.size = totalSize - sdmaSliceSize;
        sdmaSlice.push_back(sdmaSliceTmp);
        rdmaSlice.push_back(rdmaSliceTmp);
        HCCL_DEBUG("Inner data segId:%u, Orignal [offset %llu, size %llu], sdma [offset %llu, size %llu], "
                   "rdma [offset %llu, size %llu]", segsIndex, sdmaSliceOffset, totalSize, sdmaSliceTmp.offset,
                   sdmaSliceTmp.size, rdmaSliceTmp.offset, rdmaSliceTmp.size);
    }
    innerMultSlice[0] = std::make_pair(true, sdmaSlice);   // true表示使用sdma
    innerMultSlice[1] = std::make_pair(false, rdmaSlice);  // false表示rdma

    u32 commPlaneNum = innerMultSlice.size();
    u32 commIndex = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0).localRank;
    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo innerCommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);
    CHK_RET(CheckCommSize(COMM_LEVEL1_RDMA, commIndex + 1));
    SubCommInfo innerRdmaCommInfo = GetSubCommInfo(COMM_LEVEL1_RDMA, commIndex);
    for (u32 planeIndex = 0; planeIndex < commPlaneNum; planeIndex++) {
        std::vector<Slice> &singleSlice = innerMultSlice[planeIndex].second;
        SubCommInfo level1CommInfo = innerMultSlice[planeIndex].first ? innerCommInfo : innerRdmaCommInfo;
        std::unique_ptr<ExecutorBase> innerExecutor;
        if (UseInterServerNBAlgo(algType_)) {
            innerExecutor.reset(new (std::nothrow) ReduceScatterNB(dispatcher_, reduceAttr));
            HCCL_INFO("reducescatter ring: using nonuniform-bruck algo inter-server.");
        } else {
            innerExecutor.reset(new (std::nothrow) ReduceScatterRing(dispatcher_, reduceAttr));
            HCCL_INFO("reducescatter ring: using ring algo inter-server.");
        }
        CHK_SMART_PTR_NULL(innerExecutor);
        HcclResult ret = HCCL_SUCCESS;

        if (planeIndex != (commPlaneNum - 1)) {
            ret = LocalNotify::Wait(
                algResResp_->slaveStreams[planeIndex], dispatcher_, algResResp_->notifiesS2M[planeIndex], reductionOp);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("stream[%u] wait failed", planeIndex), ret);

            CHK_RET(innerExecutor->Prepare(inputMem, inputMem, scratchMem, count, dataType,
                algResResp_->slaveStreams[planeIndex], reductionOp, OUTER_BRIDGE_RANK_ID, singleSlice));

            CHK_RET(innerExecutor->RegisterProfiler(
                (level1CommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1CommInfo.localRank,
                reductionOp, HCCL_EXEC_STEP_NOT_SET, algResResp_->slaveStreams[planeIndex]));

            CHK_RET(RunTemplate(innerExecutor, level1CommInfo));
            ret = LocalNotify::Post(
                algResResp_->slaveStreams[planeIndex], dispatcher_, algResResp_->notifiesM2S[planeIndex], reductionOp);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS, HCCL_ERROR("[collAllGather]level1 stream[%u] record failed", planeIndex), ret);
            // 主环record启动从环
            ret = LocalNotify::Post(stream, dispatcher_, algResResp_->notifiesS2M[planeIndex], reductionOp);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS, HCCL_ERROR("[collAllGather]level1 stream[%u] record failed", planeIndex), ret);
        } else {
            CHK_RET(innerExecutor->Prepare(inputMem, inputMem, scratchMem, count, dataType, stream,
                reductionOp, OUTER_BRIDGE_RANK_ID, singleSlice));
            CHK_RET(innerExecutor->RegisterProfiler(
                (level1CommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1CommInfo.localRank,
                reductionOp, HCCL_EXEC_STEP_NOT_SET, stream));

            CHK_RET(RunTemplate(innerExecutor, level1CommInfo));
            for (u32 ring = 0; ring < (commPlaneNum - 1); ring++) {
                ret = LocalNotify::Wait(stream, dispatcher_, algResResp_->notifiesM2S[ring], reductionOp);
                CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("param.stream[%u] wait failed", ring), ret);
            }
        }
    }
    HCCL_INFO("Level1ReduceScatterConcurrent run success");
    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, scratchMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}

HcclResult CollCommExecutor::Level1AllReduceConcurrent(DeviceMem inputMem, DeviceMem outputMem,const u64 count,
    const HcclDataType dataType, const HcclReduceOp reductionOp, Stream stream, s32 profStage,
     std::vector<Slice> &dataSegsSlice, u32 segmentIdx, u32 commIndex, u64 hdSize, u32 syncTrans)
{
    (void)count;
    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo innerCommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);
    std::vector<std::pair<bool, Slice>> innerMultSlice;
    innerMultSlice.resize(RDMA_PLANE_NUM_IN_NPRING_DOUBLE);
    Slice sdmaSlice;
    Slice rdmaSlice;
    u64 sdmaSliceSize = ((hdSize <= HCCL_MIN_SLICE_ALIGN_910_93) || (syncTrans == MAX_SPLIT_VALUE)) ? hdSize
            : ((syncTrans * hdSize / MAX_SPLIT_VALUE) / HCCL_MIN_SLICE_ALIGN_910_93) * HCCL_MIN_SLICE_ALIGN_910_93;
    sdmaSlice.size = sdmaSliceSize;
    sdmaSlice.offset = dataSegsSlice[segmentIdx].offset;
    rdmaSlice.size = hdSize - sdmaSlice.size;
    rdmaSlice.offset = dataSegsSlice[segmentIdx].offset + sdmaSlice.size;
    HCCL_DEBUG("Level1 Total [offset:%u, size:%u], sdma [offset %llu, size %llu], rdma [offset %llu, size %llu], ",
        hdSize, sdmaSlice.offset, sdmaSlice.offset, sdmaSlice.size, rdmaSlice.offset, rdmaSlice.size);
    innerMultSlice[0] = std::make_pair(true, sdmaSlice);
    innerMultSlice[1] = std::make_pair(false, rdmaSlice);
    // SDMA和RDMA通信域
    u32 commPlaneNum = innerMultSlice.size();

    for (u32 planeIndex = 0; planeIndex < commPlaneNum; planeIndex++) {
        HcclResult ret = HCCL_SUCCESS;
        Slice &dmaSlice = innerMultSlice[planeIndex].second;
        SubCommInfo innerRdmaCommInfo = GetSubCommInfo(COMM_LEVEL1_RDMA, commIndex);
        SubCommInfo level1CommInfo = innerMultSlice[planeIndex].first ? innerCommInfo : innerRdmaCommInfo;
        DeviceMem allreduceInput = inputMem.range(dmaSlice.offset, dmaSlice.size);
        CHK_SMART_PTR_NULL(allreduceInput);
        DeviceMem allreduceOutput = outputMem.range(dmaSlice.offset, dmaSlice.size);
        CHK_SMART_PTR_NULL(allreduceOutput);
        u32 perDataSize = 0;
        CHK_RET(SalGetDataTypeSize(dataType, perDataSize));
        u64 SliceCount = dmaSlice.size / perDataSize;
        u64 reduceAttr = GetReduceAttr(allreduceInput, allreduceOutput, dataType, reductionOp);
        std::unique_ptr<ExecutorBase> innerExecutor;
        if (UseInterServerNBAlgo(algType_)) {
            innerExecutor.reset(new (std::nothrow) AllReduceNB(dispatcher_, reduceAttr));
            HCCL_INFO("allreduce ring: using nonuniform-bruck algo inter-server.");
        } else {
            innerExecutor.reset(new (std::nothrow) AllReduceRing(dispatcher_, reduceAttr));
            HCCL_INFO("allreduce ring: using ring algo inter-server.");
        }
        CHK_SMART_PTR_NULL(innerExecutor);

        if (planeIndex != (commPlaneNum - 1)) {
            ret = LocalNotify::Wait(
                algResResp_->slaveStreams[planeIndex], dispatcher_, algResResp_->notifiesS2M[planeIndex], profStage);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("stream[%u] wait failed", planeIndex), ret);

            CHK_RET(innerExecutor->Prepare(allreduceInput, allreduceOutput, allreduceOutput, SliceCount,
                dataType, algResResp_->slaveStreams[planeIndex], reductionOp, OUTER_BRIDGE_RANK_ID,
                std::vector<Slice>(0), dmaSlice.offset));

            CHK_RET(innerExecutor->RegisterProfiler(
                (level1CommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1CommInfo.localRank,
                profStage, HCCL_EXEC_STEP_NOT_SET, algResResp_->slaveStreams[planeIndex]));

            CHK_RET(RunTemplate(innerExecutor, level1CommInfo));
            ret = LocalNotify::Post(
                algResResp_->slaveStreams[planeIndex], dispatcher_, algResResp_->notifiesM2S[planeIndex], profStage);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS, HCCL_ERROR("[collAllReduce]level1 stream[%u] record failed", planeIndex), ret);
            // 主环record启动从环
            ret = LocalNotify::Post(stream, dispatcher_, algResResp_->notifiesS2M[planeIndex], profStage);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS, HCCL_ERROR("[collAllReduce]level1 stream[%u] record failed", planeIndex), ret);
        } else {
            CHK_RET(innerExecutor->Prepare(allreduceInput, allreduceOutput, allreduceOutput, SliceCount, dataType,
                stream, reductionOp, OUTER_BRIDGE_RANK_ID, std::vector<Slice>(0), dmaSlice.offset));
            CHK_RET(innerExecutor->RegisterProfiler(
                (level1CommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1CommInfo.localRank,
                profStage, HCCL_EXEC_STEP_NOT_SET, stream));

            CHK_RET(RunTemplate(innerExecutor, level1CommInfo));
            for (u32 ring = 0; ring < (commPlaneNum - 1); ring++) {
                ret = LocalNotify::Wait(stream, dispatcher_, algResResp_->notifiesM2S[ring], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("param.stream[%u] wait failed", ring), ret);
            }
        }
    }
    HCCL_INFO("Level1AllReduceConcurrent run success");
    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}

HcclResult CollCommExecutor::MultiRingMultiRootScatter(const std::string &tag, DeviceMem &inputMem,
    DeviceMem &outputMem, const u64 count, const HcclDataType dataType,
    const std::vector<std::vector<Slice>> &multRingsSliceZero, u32 root, Stream stream, const u64 baseOffset)
{
    HcclResult ret = HCCL_SUCCESS;
    u32 ringNum = multRingsSliceZero.size();
    CHK_RET(CheckCommSize(COMM_LEVEL0, ringNum));

    std::vector<std::vector<u32>> ringNics;
    CHK_RET(GetRingNics(tag, ringNics));

    for (u32 ringIndex = 0; ringIndex < ringNum; ringIndex++) {
        std::vector<Slice> singleRingSliceZero = multRingsSliceZero[ringIndex];
        CHK_PRT_RET(singleRingSliceZero.empty(),
            HCCL_ERROR("[CollCommExecutor][MultiRingMultiRootScatter]singleRingSliceZero is empty"), HCCL_E_INTERNAL);

        SubCommInfo outerRingCommInfo = GetSubCommInfo(COMM_LEVEL0, ringIndex);

        u32 rankSize = outerRingCommInfo.localRankSize;
        std::unique_ptr<ExecutorBase> executor;
        executor.reset(new (std::nothrow) MultiRootScatterRing(dispatcher_));
        CHK_SMART_PTR_NULL(executor);

        if (ringIndex != (ringNum - 1)) {
            if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) { // offline
                CHK_RET(StreamActiveManager::GetInstance(topoAttr_.deviceLogicId).StreamActive(
                    algResResp_->slaveStreams[ringIndex].ptr(), stream.ptr()));
            }
        }

        u32 rootRank = 0;
        ret = GetRankByUserRank(COMM_LEVEL0, ringIndex, root, rootRank);
        CHK_PRT_RET(ret == HCCL_E_PARA,
            HCCL_ERROR("[CollCommExecutor][MultiRingMultiRootScatter]invalid root [%u] to get userrank", root), ret);

        if (ringIndex != (ringNum - 1)) {  // 0~ringNum-2的环
            ret = LocalNotify::Wait(algResResp_->slaveStreams[ringIndex], dispatcher_,
                algResResp_->notifiesS2M[ringIndex], PROF_STAGE_0);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingMultiRootScatter]in stream[%u] wait failed", ringIndex), ret);

            ret = executor->Prepare(inputMem, outputMem, outputMem, count, dataType,
                algResResp_->slaveStreams[ringIndex], HcclReduceOp::HCCL_REDUCE_RESERVED, OUTER_BRIDGE_RANK_ID,
                singleRingSliceZero, baseOffset, ringNics[ringIndex]);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingMultiRootScatter]stream[%u],multirootscatter(ring) "\
                "prepare failed,return[%d]", ringIndex, ret), ret);

            ret = executor->RegisterProfiler(
                ((ringIndex + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) + (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) +
                outerRingCommInfo.localRank, PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET,
                algResResp_->slaveStreams[ringIndex]);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingMultiRootScatter]stream[%u], multirootscatter(ring) "\
                "register profiler failed,return[%d]", ringIndex, ret), ret);

            ret = RunTemplate(executor, outerRingCommInfo);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingMultiRootScatter]stream[%u],multirootscatter(ring) "\
                "failed,return[%d]", ringIndex, ret), ret);

            ret = LocalNotify::Post(algResResp_->slaveStreams[ringIndex], dispatcher_, algResResp_->notifiesM2S[ringIndex],
                PROF_STAGE_0);

            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingMultiRootScatter]stream[%u] record failed", ringIndex), ret);

            ret = LocalNotify::Post(stream, dispatcher_, algResResp_->notifiesS2M[ringIndex], PROF_STAGE_0);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingMultiRootScatter]stream[%u] record failed", ringIndex), ret);
        } else {  // 主环
            executor.reset(new (std::nothrow) MultiRootScatterRing(dispatcher_));
            CHK_SMART_PTR_NULL(executor);
            ret = executor->Prepare(inputMem, outputMem, outputMem, count, dataType, stream,
                HCCL_REDUCE_RESERVED, OUTER_BRIDGE_RANK_ID, singleRingSliceZero, baseOffset, ringNics[ringIndex]);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingMultiRootScatter]stream[%u],multirootscatter(ring) "\
                "prepare failed,return[%d]", ringIndex, ret), ret);

            ret = executor->RegisterProfiler(
                ((ringIndex + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) + (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID)
                + outerRingCommInfo.localRank, PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, stream);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingMultiRootScatter]stream[%u], multirootscatter(ring) "\
                "register profiler failed,return[%d]", ringIndex, ret), ret);

            ret = RunTemplate(executor, outerRingCommInfo);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingMultiRootScatter]stream[%u],multirootscatter(ring) run "\
                "failed,return[%d]", ringIndex, ret), ret);
            for (u32 ring = 0; ring < (ringNum - 1); ring++) {
                /* 等待executor执行完毕 , 当前环没有分配数据，跳过此环处理，继续下一个环 */
                ret = LocalNotify::Wait(stream, dispatcher_, algResResp_->notifiesM2S[ring], PROF_STAGE_0);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingMultiRootScatter]stream[%u] wait failed", ring), ret);
            }
        }
    }

    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}

HcclResult CollCommExecutor::MultiStreamReduceScatterMeshAtomic(const std::string &tag, DeviceMem &inputMem,
    DeviceMem &outputMem, const u64 count, const HcclDataType dataType, const HcclReduceOp reductionOp,
    const std::vector<Slice> &dataSliceVct, Stream &stream,
    const CommPlane commLevelIndex, const u64 baseOffset, HcomCollOpInfo *opInfo)
{
    u32 unitSize = SIZE_TABLE[dataType];

    u64 reduceAttr = GetReduceAttr(inputMem, outputMem, dataType, reductionOp);
    std::unique_ptr<ExecutorBase> executor;
    DeviceMem deviceOutputMem = inputMem;
    if (topoAttr_.isSingleMeshAggregation && (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) &&
        (reduceAttr & INLINE_REDUCE_BITMASK) && (opInfo != nullptr)) {
        if (((opInfo -> count) * unitSize <= HCCL_SMALL_COUNT_32_KB) &&
            (topoAttr_.deviceNumPerAggregation == DEVICE_EIGHT)) {
            deviceOutputMem = outputMem;
            executor.reset(new (std::nothrow) ReduceScatterHDStage(dispatcher_, reduceAttr, algResResp_->slaveStreams,
                algResResp_->notifiesM2S, algResResp_->notifiesS2M, topoAttr_.userRank, opInfo));
        } else {
            executor.reset(new (std::nothrow) ReduceScatterMeshDirect(dispatcher_, reduceAttr,
                algResResp_->slaveStreams, algResResp_->notifiesM2S, algResResp_->notifiesS2M,
                topoAttr_.userRank, opInfo));
        }
    } else {
        executor.reset(
            new (std::nothrow) ReduceScatterMeshAtomic(dispatcher_, reduceAttr,
            algResResp_->slaveStreams, algResResp_->notifiesM2S, algResResp_->notifiesS2M,
            topoAttr_.userRank)
        );
    }
    CHK_SMART_PTR_NULL(executor);

    CHK_RET(CheckCommSize(commLevelIndex, COMM_INDEX_0 + 1));
    const SubCommInfo subCommInfo = GetSubCommInfo(commLevelIndex, COMM_INDEX_0);
    CHK_RET(executor->Prepare(inputMem, deviceOutputMem, outputMem, count, dataType, stream, reductionOp,
        OUTER_BRIDGE_RANK_ID, dataSliceVct, baseOffset));

    CHK_RET(executor->RegisterProfiler(
        (subCommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + subCommInfo.localRank,
        PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, stream));

    CHK_RET(RunTemplate(executor, subCommInfo));

    return HCCL_SUCCESS;
}

HcclResult CollCommExecutor::MultiStreamReduceScatterMesh(const std::string &tag,
    DeviceMem inputMem, DeviceMem outputMem,
    const u64 count, const HcclDataType dataType, const HcclReduceOp reductionOp,
    const std::vector<std::vector<Slice>>& multStreamsSlice, Stream stream,
    const CommPlane commLevelIndex, const u64 baseOffset)
{
    HcclResult ret = HCCL_SUCCESS;
    u64 streamNum = multStreamsSlice.size();
    HCCL_INFO("MultiStreamReduceScatterMesh streamNum[%llu]", streamNum);
    CHK_RET(CheckCommSize(commLevelIndex, streamNum));
    const SubCommInfo zeroCommInfo = GetSubCommInfo(commLevelIndex, COMM_INDEX_0);

    u64 reduceAttr = GetReduceAttr(inputMem, outputMem, dataType, reductionOp);

    for (u32 streamIndex = 0; streamIndex < streamNum; streamIndex++) {
        std::vector<Slice> singleStreamSlice = multStreamsSlice[streamIndex];
        CHK_PRT_RET(singleStreamSlice.size() <= 0,
            HCCL_ERROR("[CollCommExecutor][MultiStreamReduceScatterMesh]singleStreamSlice is empty"),
            HCCL_E_INTERNAL);

        const SubCommInfo subCommInfo = GetSubCommInfo(commLevelIndex, streamIndex);
        u32 commIndex = subCommInfo.localRank;
        CHK_PRT_RET(commIndex >= singleStreamSlice.size(), \
            HCCL_ERROR("[CollCommExecutor][MultiStreamReduceScatterMesh]commIndex[%u] => " \
            "singleStreamSlice size[%zu]", commIndex, singleStreamSlice.size()), HCCL_E_INTERNAL);

        u32 rankSize = subCommInfo.localRankSize;
        u32 ringIndexOp = streamIndex;
        std::unique_ptr<ExecutorBase> executor;

        executor.reset(new (std::nothrow) ReduceScatterMesh(dispatcher_, reduceAttr, streamIndex));
        CHK_SMART_PTR_NULL(executor);

        if (streamIndex != (streamNum - 1)) {  // 0~ringNum-2的环
            HCCL_INFO("MultiStreamReduceScatterMesh step into subStream");
            ret = LocalNotify::Wait(algResResp_->slaveStreams[streamIndex], dispatcher_,
                algResResp_->notifiesS2M[streamIndex], PROF_STAGE_0);
            // 等待executor执行完毕
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiStreamReduceScatterMesh]stream[%u] wait failed",
                streamIndex), ret);

            ret = executor->Prepare(inputMem, inputMem, outputMem, count, dataType,
                algResResp_->slaveStreams[streamIndex], reductionOp,
                OUTER_BRIDGE_RANK_ID, singleStreamSlice, baseOffset);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiStreamReduceScatterMesh]stream[%u],reduce scatter(mesh) "\
                "prepare failed,return[%d]", streamIndex, ret), ret);

            ret = executor->RegisterProfiler(
                ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + \
                zeroCommInfo.localRank, PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET,
                algResResp_->slaveStreams[streamIndex]);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiStreamReduceScatterMesh]stream[%u],reduce scatter(mesh) "\
                "register Profiler failed,return[%d]", streamIndex, ret), ret);

            ret = RunTemplate(executor, subCommInfo);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiStreamReduceScatterMesh]stream[%u],reduce scatter(mesh) run "\
                "failed,return[%d]", streamIndex, ret), ret);

            ret  = LocalNotify::Post(algResResp_->slaveStreams[streamIndex], dispatcher_,
                algResResp_->notifiesM2S[streamIndex], PROF_STAGE_0);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiStreamReduceScatterMesh]stream[%u] record failed",
                streamIndex), ret);

            ret = LocalNotify::Post(stream, dispatcher_, algResResp_->notifiesS2M[streamIndex], PROF_STAGE_0);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiStreamReduceScatterMesh]stream[%u] record failed",
                streamIndex), ret);
        } else { // 主环
            HCCL_INFO("MultiStreamReduceScatterMesh step into mainStream");
            executor.reset(new (std::nothrow) ReduceScatterMesh(dispatcher_, reduceAttr, streamIndex));
            CHK_SMART_PTR_NULL(executor);

            ret = executor->Prepare(inputMem, inputMem, outputMem, count, dataType, stream,
                reductionOp, OUTER_BRIDGE_RANK_ID, singleStreamSlice, baseOffset);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiStreamReduceScatterMesh]stream[%u], " \
                    "reduce scatter(mesh) prepare failed, return[%d]", streamIndex, ret), ret);

            ret = executor->RegisterProfiler(
                ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + \
                zeroCommInfo.localRank, PROF_STAGE_0,
                HCCL_EXEC_STEP_NOT_SET, stream);
            CHK_PRT_RET(ret != HCCL_SUCCESS,\
                HCCL_ERROR("[CollCommExecutor][MultiStreamReduceScatterMesh]stream[%u], reduce scatter(mesh) " \
                "register Profiler failed, return[%d]", streamIndex, ret), ret);

            ret = RunTemplate(executor, subCommInfo);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiStreamReduceScatterMesh]stream[%u], " \
                    "reduce scatter(mesh) run failed, return[%d]", streamIndex, ret), ret);

            for (u32 streamIndex = 0; streamIndex < (streamNum - 1); streamIndex++) {
                //  等待executor执行完毕
                ret = LocalNotify::Wait(stream, dispatcher_, algResResp_->notifiesM2S[streamIndex], PROF_STAGE_0);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiStreamReduceScatterMesh]stream[%u] wait failed",
                        streamIndex), ret);
            }
        }
    }

    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return ret;
}

HcclResult CollCommExecutor::PrepareReduceScatterSliceData(u64 dataCount, u32 unitSize, u32 sliceNum,
    std::vector<Slice> &dataSlice)
{
    CHK_PRT_RET((sliceNum == 0), HCCL_ERROR("[CollCommExecutor][PrepareReduceScatterSliceData]sliceNum is zero."),
        HCCL_E_PARA);

    dataSlice.resize(sliceNum);
    u64 sliceSize = dataCount * unitSize;
    for (u32 i = 0; i < sliceNum; i++) {
        dataSlice[i].size = sliceSize;
        dataSlice[i].offset = (i * sliceSize);
    }
    return HCCL_SUCCESS;
}

std::vector<std::vector<u32>>  CollCommExecutor::GetRingsOrderByTopoType(u32 ranksSize, TopoType topoType,
    std::vector<u32> &nicList)
{
    std::vector<std::vector<u32>> multiRingOrder;
    if (topoType == TopoType::TOPO_TYPE_8P_RING) { // 4 ring 场景
        // 每个环的排序是按照设备物理ID进行的
        std::vector<u32> tmpOuter0 = { 0, 1, 2, 6, 5, 4, 7, 3 }; // 环0
        std::vector<u32> tmpOuter1 = { 0, 3, 7, 4, 5, 6, 2, 1 }; // 环1
        std::vector<u32> tmpOuter2 = { 0, 2, 3, 1, 5, 7, 6, 4 }; // 环2
        std::vector<u32> tmpOuter3 = { 0, 4, 6, 7, 5, 1, 3, 2 }; // 环3

        // 填充8pring 多环的comm outer 四个环的顺序
        multiRingOrder.push_back(tmpOuter0);
        multiRingOrder.push_back(tmpOuter1);
        multiRingOrder.push_back(tmpOuter2);
        multiRingOrder.push_back(tmpOuter3);
    } else if (topoType == TopoType::TOPO_TYPE_NP_DOUBLE_RING || topoType == TopoType::TOPO_TYPE_COMMON) { // 2 ring 场景
        std::vector<u32> tmpOuter0;   // 环0
        std::vector<u32> tmpOuter1;  // 环1
        std::vector<u32> rohOuter;
        if (topoMatcher_->GetExternalInputEnableRdmaSdmaConcurrent() && (topoMatcher_->CheckSdmaWithRohTopo(nicList, rohOuter)
            && workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE)) {
            tmpOuter0 = rohOuter;          // 环0, 8卡 { 0, 1, 3, 2, 4, 5, 7, 6 };
            tmpOuter1.reserve(ranksSize);  // 环1, 8卡 { 0, 6, 7, 5, 4, 2, 3, 1 };
            tmpOuter1.push_back(rohOuter[0]);
            tmpOuter1.insert(tmpOuter1.end(), rohOuter.rbegin(), rohOuter.rend() - 1);
        } else {
            tmpOuter0 = nicList;  // { 0, 1, 2, 3, 4, 5, 6, 7 };
            tmpOuter1.reserve(ranksSize);
            tmpOuter1.push_back(nicList[0]);
            tmpOuter1.insert(tmpOuter1.end(), tmpOuter0.rbegin(), tmpOuter0.rend() - 1);
        }
        // 填充 double ring 两环的comm outer的顺序
        multiRingOrder.push_back(tmpOuter0);
        multiRingOrder.push_back(tmpOuter1);
    } else { // 1 ring 场景
        std::vector<u32> tmpOuter0 = nicList; // 环0

        // 填充 single ring 单环的comm outer的顺序
        multiRingOrder.push_back(tmpOuter0);
    }
    // 打印多个环
    if (UNLIKELY(CheckDebugLogLevel())) {
        for (size_t i = 0; i < multiRingOrder.size(); i++) {
            auto ring = multiRingOrder[i];
            std::ostringstream stringRepresentation;
            for (std::vector<uint32_t>::iterator it = ring.begin(); it != ring.end(); it++) {
                stringRepresentation << *it << " ";
            }
            std::string ringString = stringRepresentation.str();
            const char *charRing = ringString.c_str();
            HCCL_DEBUG("[GetRingsOrderByTopoType] The No.%zu ring: %s", i, charRing);
        }
    }
    return multiRingOrder;
}

HcclResult CollCommExecutor::MutliSegSlicePrepare(const std::vector<Slice> &dataSegsSlice,
    std::vector<std::vector<Slice> >& mutliSegsSlices, u32 ringCount)
{
    std::vector<Slice> singleSegSlices;
    singleSegSlices.reserve(ringCount);
    for (u32 rankId = 0; rankId < dataSegsSlice.size(); rankId++) {
        Slice rankSliceTemp;
        u64 rankDataSize = dataSegsSlice[rankId].size;
        u32 ringIndex = 0;
        u64 offsetStart = dataSegsSlice[rankId].offset;
        if (rankDataSize > 0 && ringCount != 0) {
            u64 sizeTemp = (rankDataSize + ringCount - 1) / ringCount; /* 1是为了向上取整 */
            u64 sizePerRing = ExecutorBase::RoundUpWithDivisor(sizeTemp, HCCL_MIN_SLICE_ALIGN);
            u64 residueSize = rankDataSize;

            while (residueSize > 0) {
                u64 singleRingSize = sizePerRing < residueSize ? sizePerRing : residueSize;
                rankSliceTemp.size = singleRingSize;
                rankSliceTemp.offset = offsetStart + rankDataSize - residueSize;
                ringIndex++;
                if (singleRingSize == 0) {
                    HCCL_ERROR("[CollCommExecutor][MutliSegSlicePrepare]" \
                        "Multrings slices prepare: singleRingSize[%llu]",
                        singleRingSize);
                    return HCCL_E_INTERNAL;
                }
                residueSize -= singleRingSize;
                singleSegSlices.push_back(rankSliceTemp);
            }
        }
        while (ringIndex < ringCount) {
            rankSliceTemp.size = 0;
            rankSliceTemp.offset = offsetStart;
            ringIndex++;
            singleSegSlices.push_back(rankSliceTemp);
        }
        mutliSegsSlices.push_back(singleSegSlices); // rings_slice 判断大小不为 8 则异常
        singleSegSlices.clear();
    }
    return HCCL_SUCCESS;
}

HcclResult CollCommExecutor::MutliSegSlicePrepareAvoidCceRewrite(const std::vector<Slice> &dataSegsSlice,
    std::vector<std::vector<Slice> >& mutliSegsSlices, u32 ringCount) const
{
    for (u32 rankId = 0; rankId < dataSegsSlice.size(); rankId++) {
        Slice rankSliceTemp;
        std::vector<Slice> singleSegSlices;
        for (u32 ringIndex = 0; ringIndex < ringCount; ringIndex++) {
            if (ringIndex < ringCount - 1) {
                rankSliceTemp.size = 0;
                rankSliceTemp.offset = 0;
            } else {
                rankSliceTemp.size = dataSegsSlice[rankId].size;
                rankSliceTemp.offset = dataSegsSlice[rankId].offset;
            }
            singleSegSlices.push_back(rankSliceTemp);
        }
        mutliSegsSlices.push_back(singleSegSlices); // rings_slice 判断大小不为 8 则异常
    }
    return HCCL_SUCCESS;
}

void CollCommExecutor::NicSendSizeCal(const std::vector<std::vector<Slice>> &mutliSegsSlices, u32 ringCount,
    u32 chunkSize, const std::vector<u32> &nicList, const std::string &tag)
{
    // 计算每个网口最终会发送的数据量大小
    std::vector<u64> sizeList;
    sizeList.reserve(nicList.size());
    for (u32 nicIdx = 0; nicIdx < nicList.size(); nicIdx++) {
        u64 tempSize = 0;
        for (u32 chunkIdx = 0; chunkIdx < chunkSize; chunkIdx++) {
            for (u32 ringIdx = 0; ringIdx < ringCount; ringIdx++) {
                tempSize += mutliSegsSlices[nicIdx * chunkSize + chunkIdx][ringIdx].size;
            }
        }
        sizeList.push_back(tempSize);
    }
    SetNicSendSize(tag, sizeList);
}

std::vector<std::vector<Slice> > CollCommExecutor::PrepareMultiRingSliceRS(const std::vector<Slice> &dataSegsSlice,
    const std::string &tag, bool avoidCceRewrite, std::vector<u32> nicList)
{
    // get ranksSize
    u32 ranksSize = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0).localRankSize;
    // 获取每个ring上设备的排布顺序，顺序均为deviceID
    sort(nicList.begin(), nicList.end());
    std::vector<std::vector<u32> > multiRingsOrder = GetRingsOrderByTopoType(ranksSize, topoType_, nicList);
    std::vector<std::vector<Slice> > mutliRingsSlices;
    std::vector<std::vector<Slice> > mutliSegsSlices;
    u32 ringCount = multiRingsOrder.size();
    // 单环场景不应该走入此流程，需要在函数外校验
    CHK_PRT_RET(ringCount <= 1, HCCL_ERROR("[CollCommExecutor][PrepareMultiRingSlice] ringCount[%u] <= 1",
        ringCount), mutliRingsSlices);

    u32 ringRanks = multiRingsOrder[0].size(); // 获取单个 ring 上设备的数量

    // 将数每块据切分为 ringCount 份
    HcclResult ret;
    mutliSegsSlices.reserve(dataSegsSlice.size());

    // ===============================================
    // if (avoidCceRewrite) {
    //     ret = MutliSegSlicePrepareAvoidCceRewrite(dataSegsSlice, mutliSegsSlices, ringCount);
    // } else {
    //     ret = MutliSegSlicePrepare(dataSegsSlice, mutliSegsSlices, ringCount);
    // }
    // if (ret != HCCL_SUCCESS) {
    //     return mutliRingsSlices;
    // }
    // ===============================================

    for (u32 sliceIdx = 0; sliceIdx < dataSegsSlice.size(); sliceIdx++){
        std::vector<Slice> singleSegSlices;
        for (u32 i = 0; i < 2; i++){
            singleSegSlices.emplace_back(dataSegsSlice[sliceIdx]);
        }
        multiSegsSlices.emplace_back(singleSegSlices);
    }


    // ===============================================
    u32 chunkSize = ringRanks / nicList.size();
    (void) NicSendSizeCal(mutliSegsSlices, ringCount, chunkSize, nicList, tag);
    std::vector<std::vector<u32>> ringRankList;
    std::vector<Slice> singleRingSlices;
    std::vector<u32> rankList;

    ringRankList.reserve(ringCount);
    singleRingSlices.reserve(ringRanks);
    rankList.reserve(ringRanks);

    for (u32 ringIndex = 0; ringIndex < ringCount; ringIndex++) {
        for (u32 segsIndex = 0; segsIndex < ringRanks; segsIndex++) {
            u32 deviceIdx = multiRingsOrder[ringIndex][segsIndex];
            std::vector<u32>::iterator iterRank = std::find(nicList.begin(), nicList.end(), deviceIdx);
            if (iterRank != nicList.end()) {
                rankList.push_back(segsIndex);
                u32 nicPosition = distance(nicList.begin(), iterRank);
                for (u32 chunkIdx = 0; chunkIdx < chunkSize; chunkIdx++) {
                    Slice tempSlice = mutliSegsSlices[nicPosition * chunkSize + chunkIdx][ringIndex];
                    singleRingSlices.push_back(tempSlice);
                }
            }
        }
        mutliRingsSlices.push_back(singleRingSlices);
        ringRankList.push_back(rankList);
        singleRingSlices.clear();
        rankList.clear();
    }

    ret = SetRingNics(tag, ringRankList);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[Prepare][MultiRingSlice]set nics in ring failed, ret[%u]", ret);
        std::vector<std::vector<Slice> > emptySlice;
        return emptySlice;
    }
    return mutliRingsSlices;
}


std::vector<std::vector<Slice> > CollCommExecutor::PrepareMultiRingSlice(const std::vector<Slice> &dataSegsSlice,
    const std::string &tag, bool avoidCceRewrite, std::vector<u32> nicList)
{
    // get ranksSize
    u32 ranksSize = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0).localRankSize;
    // 获取每个ring上设备的排布顺序，顺序均为deviceID
    sort(nicList.begin(), nicList.end());
    std::vector<std::vector<u32> > multiRingsOrder = GetRingsOrderByTopoType(ranksSize, topoType_, nicList);
    std::vector<std::vector<Slice> > mutliRingsSlices;
    std::vector<std::vector<Slice> > mutliSegsSlices;
    u32 ringCount = multiRingsOrder.size();
    // 单环场景不应该走入此流程，需要在函数外校验
    CHK_PRT_RET(ringCount <= 1, HCCL_ERROR("[CollCommExecutor][PrepareMultiRingSlice] ringCount[%u] <= 1",
        ringCount), mutliRingsSlices);

    u32 ringRanks = multiRingsOrder[0].size(); // 获取单个 ring 上设备的数量

    // 将数每块据切分为 ringCount 份
    HcclResult ret;
    mutliSegsSlices.reserve(dataSegsSlice.size());
    if (avoidCceRewrite) {
        ret = MutliSegSlicePrepareAvoidCceRewrite(dataSegsSlice, mutliSegsSlices, ringCount);
    } else {
        ret = MutliSegSlicePrepare(dataSegsSlice, mutliSegsSlices, ringCount);
    }
    if (ret != HCCL_SUCCESS) {
        return mutliRingsSlices;
    }
    u32 chunkSize = ringRanks / nicList.size();
    (void) NicSendSizeCal(mutliSegsSlices, ringCount, chunkSize, nicList, tag);
    std::vector<std::vector<u32>> ringRankList;
    std::vector<Slice> singleRingSlices;
    std::vector<u32> rankList;

    ringRankList.reserve(ringCount);
    singleRingSlices.reserve(ringRanks);
    rankList.reserve(ringRanks);

    for (u32 ringIndex = 0; ringIndex < ringCount; ringIndex++) {
        for (u32 segsIndex = 0; segsIndex < ringRanks; segsIndex++) {
            u32 deviceIdx = multiRingsOrder[ringIndex][segsIndex];
            std::vector<u32>::iterator iterRank = std::find(nicList.begin(), nicList.end(), deviceIdx);
            if (iterRank != nicList.end()) {
                rankList.push_back(segsIndex);
                u32 nicPosition = distance(nicList.begin(), iterRank);
                for (u32 chunkIdx = 0; chunkIdx < chunkSize; chunkIdx++) {
                    Slice tempSlice = mutliSegsSlices[nicPosition * chunkSize + chunkIdx][ringIndex];
                    singleRingSlices.push_back(tempSlice);
                }
            }
        }
        mutliRingsSlices.push_back(singleRingSlices);
        ringRankList.push_back(rankList);
        singleRingSlices.clear();
        rankList.clear();
    }

    ret = SetRingNics(tag, ringRankList);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[Prepare][MultiRingSlice]set nics in ring failed, ret[%u]", ret);
        std::vector<std::vector<Slice> > emptySlice;
        return emptySlice;
    }
    return mutliRingsSlices;
}

std::vector<std::vector<Slice>> CollCommExecutor::ASYMMultiRingSlicePrepare(u32 ringNum, u32 sliceNum,
    bool useInlineReduce, DeviceMem& outputMem, std::vector<Slice>& dataSegsSlice, const std::string &tag)
{
    std::vector<std::vector<Slice>> multiStreamSlice;
    u64 outputMenSize = outputMem.size();
    dataSegsSlice.clear();
    Slice sliceTemp;
    for (u32 i = 0; i < sliceNum; i++) {    // 根据数据量算每个环上数据的偏移和大小
        sliceTemp.size = outputMenSize;
        sliceTemp.offset = outputMenSize * i;
        dataSegsSlice.push_back(sliceTemp);
    }

    // 再将每个 slice 划分为 ringNum 份
    if (ringNum == OUTER_PLANE_NUM_IN_8PRING) {
        // 双环数据相同
            for(int i=0;i<2;++i){
                multiStreamSlice.push_back(dataSegsSlice);
            }

            std::vector<Slice>& secondVector = multiStreamSlice[1];
            size_t n = secondVector.size();
            for (size_t i = 1; i < n / 2; ++i) {
                std::swap(secondVector[i], secondVector[n - i]);
            }
    } else if (ringNum == OUTER_PLANE_NUM_IN_NPRING_DOUBLE) {
        // 双环数据相同
            for(int i=0;i<2;++i){
                multiStreamSlice.push_back(dataSegsSlice);
            }

            std::vector<Slice>& secondVector = multiStreamSlice[1];
            size_t n = secondVector.size();
            for (size_t i = 1; i < n / 2; ++i) {
                std::swap(secondVector[i], secondVector[n - i]);
            }
    } else {
        multiStreamSlice.push_back(dataSegsSlice);
    }

    return multiStreamSlice;
}

u64 CollCommExecutor::GetReduceAttr(DeviceMem &inputMem, DeviceMem &outputMem, HcclDataType dataType, HcclReduceOp op)
{
    u64 reduceAttr = 0;
    bool isInlineReduce = IsSupportSDMAReduce(inputMem.ptr(), outputMem.ptr(), dataType, op);
    if (isInlineReduce && algoAttr_.inlineReduceSwitchOn) {
        SalSetBitOne(reduceAttr, ATTR_POS_INLINE_REDUCE);
    }

    bool isRdmaReduce = IsOverFlowInfNanMode() && IsSupportRDMAReduce(dataType, op);
    if (isRdmaReduce) {
        SalSetBitOne(reduceAttr, ATTR_POS_SUPPORT_RDMA_REDUCE);
    }

    return reduceAttr;
}

HcclResult CollCommExecutor::CalUserMemSlices(const HcclDataType dataType, const HcomCollOpInfo *opInfo,
                                              const std::vector<Slice> &singleRingSliceZero, u32 ringIndex,
                                              const std::vector<std::vector<u32>> &multiRingsOrder,
                                              std::vector<Slice>                  &userMemSlices)
{
    if (opInfo == nullptr || opInfo->inputAddr == nullptr || opInfo->outputAddr == nullptr) {
        // 910_93场景下，allreduce算子的userMem上的slice信息
        userMemSlices = singleRingSliceZero;
        return HCCL_SUCCESS;
    }
    // 910_93场景下，reduce scatter和all gather算子的userMem上的slice信息
    std::vector<u32> ring0 = multiRingsOrder[0];
    for (u32 sliceIdx = 0; sliceIdx < singleRingSliceZero.size(); sliceIdx++) {
        Slice userMemSlice;
        u32 deviceId;
        if (ringIndex >= SLICES_FACTOR){
            deviceId = multiRingsOrder[ringIndex % SLICES_FACTOR][sliceIdx];
        } else {
            deviceId = multiRingsOrder[ringIndex][sliceIdx];
        }

        u32 pos = distance(ring0.begin(), find(ring0.begin(), ring0.end(), deviceId));
        userMemSlice.offset = pos * opInfo->count * SIZE_TABLE[dataType]
                                + singleRingSliceZero[0].offset;
        userMemSlice.size = singleRingSliceZero[sliceIdx].size;
        userMemSlices.push_back(userMemSlice);
        HCCL_DEBUG(
            "[CollCommExecutor][CalUserMemSlices] Push back userMemSlice offset[%llu], size[%llu] at rank[%u]",
            userMemSlice.offset, userMemSlice.size, topoAttr_.userRank);
    }
    return HCCL_SUCCESS;
}

HcclResult CollCommExecutor::GetRankOrder(const std::vector<std::vector<u32>> &multiRingsOrder, u32 ringIndex,
    std::vector<u32> &rankOrder)
{
    std::vector<u32> ring0 = multiRingsOrder[0];
    std::vector<u32> ringOrder = multiRingsOrder[ringIndex];
    for (u32 i = 0; i < ringOrder.size(); i++) {
        u32 deviceId = ringOrder[i];
        u32 pos = distance(ring0.begin(), find(ring0.begin(), ring0.end(), deviceId));
        rankOrder.push_back(pos);
    }
    return HCCL_SUCCESS;
}

u32 CollCommExecutor::RefreshCommIdx(u32 commIndex, std::vector<u32> nicList, u32 devicePhyId)
{
    if (topoMatcher_->GetExternalInputEnableRdmaSdmaConcurrent() && CheckRankNeighbors(nicList)) {
        std::vector<u32>::iterator iterRank = std::find(nicList.begin(), nicList.end(), devicePhyId);
        // 按照实际topo寻找对应的rankID,即commIndex
        if (iterRank != nicList.end()) {
            u32 nicPosition = distance(nicList.begin(), iterRank);
            if (commIndex != nicPosition) {
                HCCL_DEBUG(
                    "[RefreshCommIdx] old commIndex %u, new commIndex %u", commIndex, nicPosition);
                commIndex = nicPosition;
            }
        }
    }
    return commIndex;
}

HcclResult CollCommExecutor::MultiRingScatter(const std::string &tag, DeviceMem inputMem, DeviceMem outputMem,
    const u64 count, const HcclDataType dataType, const std::vector<std::vector<Slice> > multRingsSliceZero,
    u32 root, Stream stream, const HcomCollOpInfo *opInfo, const u64 baseOffset)
{
    HcclResult ret = HCCL_SUCCESS;
    u32 ringNum = multRingsSliceZero.size();

    CHK_RET(CheckCommSize(COMM_LEVEL0, ringNum));

    std::vector<std::vector<u32>> ringNics;
    CHK_RET(GetRingNics(tag, ringNics));

    // 拿到ring环映射关系
    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    auto nicList = topoAttr_.nicList;
    std::vector<std::vector<u32>> multiRingsOrder = GetRingsOrderByTopoType(outerCommInfo.localRankSize, topoType_, nicList);

    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    for (u32 ringIndex = 0; ringIndex < ringNum; ringIndex++) {
        std::vector<Slice> singleRingSliceZero = multRingsSliceZero[ringIndex];
        CHK_PRT_RET(singleRingSliceZero.empty(),
            HCCL_ERROR("[CollCommExecutor][MultiRingScatter]singleRingSliceZero is empty"), HCCL_E_INTERNAL);

        // 生成userMemIn_上对应的slices
        std::vector<Slice> userMemInputSlices;
        CHK_RET(
            CalUserMemSlices(dataType, opInfo, singleRingSliceZero, ringIndex, multiRingsOrder, userMemInputSlices));
        std::vector<u32> rankOrder;
        CHK_RET(GetRankOrder(multiRingsOrder, ringIndex, rankOrder));
        SubCommInfo outerRingCommInfo = GetSubCommInfo(COMM_LEVEL0, ringIndex);
        u32 rankSize = outerRingCommInfo.localRankSize;

        std::vector<Stream> subStreamsInOneRing;
        std::vector<std::shared_ptr<LocalNotify>> mainSignalsInOneRing;
        std::vector<std::shared_ptr<LocalNotify>> subSignalsInOneRing;
        std::unique_ptr<ExecutorBase> executor;
        if (opInfo != nullptr) {
            CHK_RET(GetSubStreamInfoOnOneRing(ringIndex, subStreamsInOneRing, mainSignalsInOneRing,
                                              subSignalsInOneRing));
            executor.reset(new (std::nothrow) ScatterRingConcurrentDirect(
                dispatcher_, opInfo, topoAttr_.userRank, subStreamsInOneRing,
                mainSignalsInOneRing, subSignalsInOneRing, rankOrder, userMemInputSlices));
        } else {
            executor.reset(new (std::nothrow) ScatterRing(dispatcher_));
        }
        CHK_SMART_PTR_NULL(executor);

        if (ringIndex != (ringNum - 1)) {
            if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) { // offline
                ret = StreamActiveManager::GetInstance(topoAttr_.deviceLogicId).StreamActive(
                    algResResp_->slaveStreams[ringIndex].ptr(), stream.ptr());
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingScatter]stream[%u],active stream failed", ringIndex), ret);
            }
        }

        u32 rootRank = 0;
        ret = GetRankByUserRank(COMM_LEVEL0, ringIndex, root, rootRank);
        CHK_PRT_RET(ret == HCCL_E_PARA,
            HCCL_ERROR("[CollCommExecutor][MultiRingScatter]invalid root [%u] to get userrank", root), ret);

        if (ret == HCCL_SUCCESS) {
            if (ringIndex != (ringNum - 1)) {  // 0~ringNum-2的环
                ret = LocalNotify::Wait(algResResp_->slaveStreams[ringIndex], dispatcher_,
                    algResResp_->notifiesS2M[ringIndex], PROF_STAGE_0);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingScatter]in stream[%u] wait failed", ringIndex), ret);

                ret = executor->Prepare(inputMem, inputMem, outputMem, count, dataType,
                    algResResp_->slaveStreams[ringIndex], HCCL_REDUCE_RESERVED, rootRank, singleRingSliceZero,
                    baseOffset, ringNics[ringIndex]);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingScatter]stream[%u],scatter(ring) prepare failed, "\
                    "return[%d]", ringIndex, ret), ret);

                ret = executor->RegisterProfiler(((ringIndex + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                    (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + outerRingCommInfo.localRank,
                    PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, algResResp_->slaveStreams[ringIndex]);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingScatter]stream[%u], scatter(ring) register profiler "\
                    "failed,return[%d]", ringIndex, ret), ret);

                ret = RunTemplate(executor, outerRingCommInfo);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingScatter]stream[%u],scatter(ring) run failed, "\
                    "return[%d]", ringIndex, ret), ret);

                ret = LocalNotify::Post(algResResp_->slaveStreams[ringIndex], dispatcher_,
                    algResResp_->notifiesM2S[ringIndex], PROF_STAGE_0);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingScatter]stream[%u] record failed", ringIndex), ret);
                /* 主环record启动从环 */
                ret = LocalNotify::Post(stream, dispatcher_, algResResp_->notifiesS2M[ringIndex], PROF_STAGE_0);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingScatter]stream[%u] record failed", ringIndex), ret);
            } else {  // 主环
                std::unique_ptr<ExecutorBase> executor;
                if (opInfo != nullptr) {
                    executor.reset(new (std::nothrow) ScatterRingConcurrentDirect(
                        dispatcher_, opInfo, topoAttr_.userRank, subStreamsInOneRing, mainSignalsInOneRing,
                        subSignalsInOneRing, rankOrder, userMemInputSlices));
                } else {
                    executor.reset(new (std::nothrow) ScatterRing(dispatcher_));
                }
                CHK_SMART_PTR_NULL(executor);
                ret = executor->Prepare(inputMem, inputMem, outputMem, count, dataType, stream,
                    HCCL_REDUCE_RESERVED, rootRank, singleRingSliceZero, baseOffset, ringNics[ringIndex]);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingScatter]stream[%u],scatter(ring) prepare failed, "\
                    "return[%d]", ringIndex, ret), ret);
                ret = executor->RegisterProfiler(((ringIndex + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                    (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + outerRingCommInfo.localRank,
                    PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, stream);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingScatter]stream[%u], scatter(ring) register profiler "\
                    "failed,return[%d]", ringIndex, ret), ret);

                ret = RunTemplate(executor, outerRingCommInfo);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingScatter]stream[%u],scatter(ring) run failed, "\
                    "return[%d]", ringIndex, ret), ret);

                for (u32 ring = 0; ring < (ringNum - 1); ring++) {
                    /* 等待executor执行完毕 , 当前环没有分配数据，跳过此环处理，继续下一个环 */
                    ret = LocalNotify::Wait(stream, dispatcher_, algResResp_->notifiesM2S[ring], PROF_STAGE_0);
                    CHK_PRT_RET(ret != HCCL_SUCCESS,
                        HCCL_ERROR("[CollCommExecutor][MultiRingScatter]stream[%u] wait failed", ring), ret);
                }
            }
        }
    }

    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}

HcclResult CollCommExecutor::SetRingNics(const std::string &tag, const std::vector<std::vector<u32>> &ringNics)
{
    std::unique_lock<std::mutex> lock(ringNicListLock_);
    ringNicList_[tag] = ringNics;
    return HCCL_SUCCESS;
}
HcclResult CollCommExecutor::GetRingNics(const std::string &tag, std::vector<std::vector<u32>> &ringNics)
{
    std::unique_lock<std::mutex> lock(ringNicListLock_);
    auto iterRingNic = ringNicList_.find(tag);
    if (iterRingNic == ringNicList_.end()) {
        ringNics = {{0, 1, 2, 3, 4, 5, 6, 7}};
    } else {
        ringNics = iterRingNic->second;
    }
    return HCCL_SUCCESS;
}
HcclResult CollCommExecutor::SetNicSendSize(const std::string &tag, std::vector<u64> &sizeList)
{
    std::unique_lock<std::mutex> lock(nicSendSizeListLock_);
    nicSendSizeList_[tag] = sizeList;
    return HCCL_SUCCESS;
}
HcclResult CollCommExecutor::PrepareInnerCommInfo(u32 &segmentIdx, u32 &commIndex, u64 &hdSize,
                                                  const SubCommInfo &commInfo,
                                                  const std::vector<std::vector<Slice>> &multRingsSliceZero,
                                                  const std::string &tag)
{
    segmentIdx = topoAttr_.devicePhyId;
    commIndex = topoAttr_.devicePhyId;
    CHK_PRT_RET(multRingsSliceZero.empty(), HCCL_ERROR("[Prepare][InnerCommInfo]sicle map is empty"), HCCL_E_PARA);
    if (multRingsSliceZero.size() > 1) {
        std::vector<u32>::const_iterator iterNic = std::find(topoAttr_.nicList.begin(),
                                                             topoAttr_.nicList.end(), topoAttr_.devicePhyId);
        if (iterNic != topoAttr_.nicList.end()) {                          // 如果当前rank为通信网口
            u32 nicIdx = std::distance(topoAttr_.nicList.begin(), iterNic);
            std::unique_lock<std::mutex> lock(nicSendSizeListLock_);
            auto iter = nicSendSizeList_.find(tag);
            CHK_PRT_RET(iter == nicSendSizeList_.end(), HCCL_ERROR("[Prepare][InnerCommInfo]find tag[%s] in "\
                "nicSendSizeList_ failed", tag.c_str()), HCCL_E_INTERNAL);
            CHK_PRT_RET(nicIdx >= iter->second.size(), HCCL_ERROR("[Prepare][InnerCommInfo]tag[%s] nicIdx[%u] "\
                "invaild, expect less than %zu", tag.c_str(), nicIdx, iter->second.size()), HCCL_E_INTERNAL);
            hdSize = iter->second[nicIdx];                    // 通过nicSendSizeList_得到该网口传输数据量
            u32 ringRanks = multRingsSliceZero[0].size(); // 获取单个 ring 上设备的数量
            segmentIdx = ringRanks / topoAttr_.nicList.size() * nicIdx; // 通过网口位置得到该网口传输数据的起始位置
            if (topoAttr_.deviceType == DevType::DEV_TYPE_910_93) {
                commIndex = segmentIdx;
            }
        } else {                                                  // 如果当前rank不是通信网口，则不发送数据
            hdSize = 0;
        }
    } else if (multRingsSliceZero.size() == 1) {
        segmentIdx = commInfo.localRank; // 针对0、4device下
        CHK_PRT_RET(segmentIdx >= multRingsSliceZero[0].size(), HCCL_ERROR("[Prepare][InnerCommInfo]index is out of "\
            "range. Idx[%u] Slice size[%zu]", segmentIdx, multRingsSliceZero[0].size()), HCCL_E_PARA);
        hdSize = multRingsSliceZero[0][segmentIdx].size;
        commIndex = segmentIdx;
    } else {
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}
}