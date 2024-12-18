/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "coll_aligned_reduce_scatter_asym_double_ring_executor.h"

namespace hccl {
CollAlignedReduceScatterAsymDoubleRingExecutor::CollAlignedReduceScatterAsymDoubleRingExecutor(
    const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollReduceScatterRingFor91093Executor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
}

HcclResult CollAlignedReduceScatterAsymDoubleRingExecutor::DoubleRingReduceScatter(
    const std::string &tag, DeviceMem inputMem, DeviceMem outputMem,
    const u64 count, const HcclDataType dataType, const HcclReduceOp reductionOp,
    const std::vector<std::vector<Slice> > multRingsSliceZero, Stream stream, s32 profStage,
    const u64 baseOffset, const HcomCollOpInfo *opInfo,
    const std::vector<std::vector<Slice>> multRingsUserMemSlice, const bool retryEnable)
{
    (void)tag;
    HCCL_INFO(
        "[CollAlignedReduceScatterAsymDoubleRingExecutor][DoubleRingReduceScatter] DoubleRingReduceScatter starts");
    HcclResult ret = HCCL_SUCCESS;
    u32 ringNum = multRingsSliceZero.size();
    CHK_RET(CheckCommSize(COMM_LEVEL0, ringNum));

    // 拿到ring环映射关系
    SubCommInfo outerZeroCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    auto nicList = topoAttr_.nicList;
    std::vector<std::vector<u32>> multiRingsOrder =
        GetRingsOrderByTopoType(outerZeroCommInfo.localRankSize, topoType_, nicList);

    u64 reduceAttr = GetReduceAttr(inputMem, outputMem, dataType, reductionOp);

    SubCommInfo outerRingCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    // 生成两个ring上的userMemIn_上对应的slices
    std::vector<std::vector<Slice>> userMemInputSlicesOfDoubleRing;
    CHK_RET(CollectMultiRingsUserMemSlices(ringNum, dataType,
        opInfo, multRingsSliceZero,
        multiRingsOrder, multRingsUserMemSlice,
        userMemInputSlicesOfDoubleRing));
    // 生成两个ring上的rankOrder
    std::vector<std::vector<u32>> rankOrders;
    CHK_RET(CollectMultiRingsRankOrder(ringNum, multiRingsOrder, rankOrders));
    // 初始化executor
    std::unique_ptr<ExecutorBase> executor;
    executor.reset(new (std::nothrow) AlignedReduceScatterDoubleRing(
        dispatcher_, reduceAttr, opInfo, topoAttr_.userRank, algResResp_->slaveStreams,
        algResResp_->notifiesM2S, algResResp_->notifiesS2M, rankOrders, userMemInputSlicesOfDoubleRing));
    CHK_SMART_PTR_NULL(executor);
    ret = executor->Prepare(inputMem, inputMem, outputMem, count, dataType, stream, multRingsSliceZero,
        reductionOp, OUTER_BRIDGE_RANK_ID, baseOffset, retryEnable);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAlignedReduceScatterAsymDoubleRingExecutor][DoubleRingReduceScatter] Double ring reduce scatter failed"
        "failed,return[%d]", ret), ret);
    u32 ringIndexOp = COMM_INDEX_0;
    u32 rankSize = outerRingCommInfo.localRankSize;
    ret = executor->RegisterProfiler(
        ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
        (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + outerRingCommInfo.localRank,
        profStage, HCCL_EXEC_STEP_NOT_SET, stream);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAlignedReduceScatterAsymDoubleRingExecutor][DoubleRingReduceScatter] Double ring reduce scatter failed "
        "failed,return[%d]", ret), ret);

    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    ret = RunTemplate(executor, outerRingCommInfo);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAlignedReduceScatterAsymDoubleRingExecutor][DoubleRingReduceScatter] Double ring reduce scatter failed "
        "failed,return[%d]", ret), ret);

    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}

HcclResult CollAlignedReduceScatterAsymDoubleRingExecutor::RunIntraSeverReduceScatter(
    const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
    const u64 count, const HcclDataType &dataType, const HcclReduceOp &reductionOp,
    const std::vector<std::vector<Slice>> &multRingsSliceZero, const Stream &stream, s32 profStage,
    const u64 baseOffset, const HcomCollOpInfo *opInfo,
    const std::vector<std::vector<Slice>> &multRingsUserMemSlice, const bool retryEnable)
{
    CHK_RET(DoubleRingReduceScatter(tag, inputMem, outputMem, count, dataType, reductionOp,
        multRingsSliceZero, stream, profStage, baseOffset, opInfo, multRingsUserMemSlice, retryEnable));
    return HCCL_SUCCESS;
}

void CollAlignedReduceScatterAsymDoubleRingExecutor::FillMultiRingSlice(
    const ExecMem &execMem,
    const std::vector<std::vector<Slice>> &multiStreamSlice,
    u32 sliceNum, u32 innerRankSize, u32 level2RankSize,
    const u32 ringIndex, std::vector<Slice> &dataSlice)
{
    for (u32 level0Idx = 0; level0Idx < sliceNum; level0Idx++) {
        Slice sliceTemp;
        for (u32 level2Idx = 0; level2Idx < level2RankSize; level2Idx++) {
            for (u32 level1Idx = 0; level1Idx < innerRankSize; level1Idx++) {
                sliceTemp.size = multiStreamSlice[ringIndex][level0Idx].size;
                sliceTemp.offset = multiStreamSlice[ringIndex][level0Idx].offset +
                    level1Idx * sliceNum * execMem.outputMem.size() +
                    level2Idx * sliceNum * innerRankSize * execMem.outputMem.size();
                dataSlice.push_back(sliceTemp);
                HCCL_DEBUG("rank[%u] sliceTemp.size[%zu]，sliceTemp.offset[%llu]", topoAttr_.userRank,
                    sliceTemp.size, sliceTemp.offset);
            }
        }
    }
}

void CollAlignedReduceScatterAsymDoubleRingExecutor::CalLevel0DataSegsSlice(
    const ExecMem &execMem,
    const std::vector<std::vector<Slice>> &multiStreamSlice,
    u32 sliceNum, u32 innerRankSize, u32 level2RankSize,
    std::vector<std::vector<Slice>> &level0DataSegsSlice)
{
    for (u32 ringIndex = 0; ringIndex < multiStreamSlice.size(); ringIndex++) {
        std::vector<Slice> dataSlice;
        FillMultiRingSlice(execMem, multiStreamSlice, sliceNum, innerRankSize, level2RankSize, ringIndex, dataSlice);
        level0DataSegsSlice.push_back(dataSlice);
    }
}

HcclResult CollAlignedReduceScatterAsymDoubleRingExecutor::CalLevel1DataSegsSlice(
    const ExecMem &execMem, const u32 &commIndex,
    u32 sliceNum, u32 innerRankSize, u32 level2RankSize,
    std::vector<Slice> &level1DataSegsSlice)
{
    for (u32 i = 0; i < innerRankSize; i++) {
        Slice sliceTemp;
        u32 level1UserRank;
        CHK_RET(GetUserRankByRank(COMM_LEVEL1, commIndex, i, level1UserRank));
        if (level2RankSize <= 1) {
            sliceTemp.size = execMem.outputMem.size();
            sliceTemp.offset = level1UserRank * execMem.outputMem.size();
            level1DataSegsSlice.push_back(sliceTemp);
            HCCL_DEBUG("rank[%u], level1DataSegsSlice[%u].offset=%llu, size=[%llu]", topoAttr_.userRank, i,
                sliceTemp.offset, sliceTemp.size);
        } else {
            for (u32 level2Idx = 0; level2Idx < level2RankSize; level2Idx++) {
                sliceTemp.size = execMem.outputMem.size();
                sliceTemp.offset = (level1UserRank % (innerRankSize * sliceNum)) * execMem.outputMem.size() +
                        level2Idx * sliceNum * innerRankSize * execMem.outputMem.size();
                level1DataSegsSlice.push_back(sliceTemp);
                HCCL_DEBUG("rank[%u], level1DataSegsSlice[%u].offset=%llu, size=[%llu]", topoAttr_.userRank, i,
                    sliceTemp.offset, sliceTemp.size);
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CollAlignedReduceScatterAsymDoubleRingExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_INFO("[CollReduceScatterRingFor91093Executor][KernelRun] The ReduceScatterDoubleRingExecutor starts.");
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(param.DataDes.dataType, perDataSize));

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    u32 ringNum;
    if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
        ringNum = OUTER_PLANE_NUM_IN_NPRING_DOUBLE;
    } else {
        ringNum = OUTER_PLANE_NUM_IN_NPRING_SINGLE;
    }

    u32 sliceNum = outerCommInfo.localRankSize;
    Slice sliceTemp;
    u32 commIndex = outerCommInfo.localRank;
    commIndex = RefreshCommIdx(commIndex, topoAttr_.nicList, topoAttr_.devicePhyId);

    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo innerCommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    CHK_RET(CheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1));
    SubCommInfo level2CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);
    u32 level2RankSize = level2CommInfo.localRankSize;

    std::vector<Slice> dataSegsSlice;   // 数据分成ranksize份，每份的起始偏移和大小
    std::vector<std::vector<Slice>> multiStreamSlice; // 每个stream使用的数据基于用户buffer的偏移

    // 节点内reduce scatter
    CHK_RET(ActiveSlaveStreams(param.stream));
    u32 innerRankSize = innerCommInfo.localRankSize;

    // 计算slice
    std::vector<std::vector<Slice>> level0DataSegsSlice;
    bool useInlineRduce = false;
    bool isInlineReduce = IsSupportSDMAReduce(execMem.inputMem.ptr(), execMem.scratchMem.ptr(), param.DataDes.dataType,
        param.reduceType);
    useInlineRduce = isInlineReduce && algoAttr_.inlineReduceSwitchOn;
    multiStreamSlice = ReduceScatterRingSlicePrepare(ringNum, sliceNum, useInlineRduce, execMem.outputMem,
        dataSegsSlice, param.tag);

    printf("dataSegsSlice.size(): %d\n", dataSegsSlice.size());
    for (size_t i = 0; i < dataSegsSlice.size(); ++i) {
            const Slice& slice = dataSegsSlice[i];
            std::cout << "  Slice " << i << " - Offset: " << slice.offset << ", Size: " << slice.size << " bytes\n";
    }

    printf("multiStreamSlice.size(): %d\n", multiStreamSlice.size());
    printf("multiStreamSlice[0].size(): %d\n", multiStreamSlice[0].size());
    for (size_t i = 0; i < multiStreamSlice.size(); ++i) {
        std::cout << "Stream " << i << ":\n";
        for (size_t j = 0; j < multiStreamSlice[i].size(); ++j) {
            const Slice& slice = multiStreamSlice[i][j];
            std::cout << "  Slice " << j << " - Offset: " << slice.offset << ", Size: " << slice.size << " bytes\n";
        }
    }

    CalLevel0DataSegsSlice(execMem, multiStreamSlice, sliceNum, innerRankSize, level2RankSize, level0DataSegsSlice);

    std::vector<std::vector<Slice>> multRingsUserMemSlice;

    HcomCollOpInfo opInfo = {"", execMem.inputPtr, execMem.outputPtr, param.DataDes.count, param.DataDes.dataType,
        param.root, param.reduceType};
    HCCL_DEBUG("[CollReduceScatterRingFor91093Executor][KernelRun] execMem.inputPtr[%p], execMem.outputPtr[%p], execMem.inputMem[%p], execMem.outputMem[%p]",
        execMem.inputPtr, execMem.outputPtr, execMem.inputMem.ptr(), execMem.outputMem.ptr());
    HcomCollOpInfo *opInfoPtr = nullptr;
    if (DMAReduceFlag_) {
        opInfoPtr = &opInfo;
    }

    if (opInfoPtr == nullptr &&
        (!(topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING &&
        (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB || param.retryEnable)))) {
        multRingsUserMemSlice = level0DataSegsSlice;
    } else {
        for (u32 ringIndex = 0; ringIndex < level0DataSegsSlice.size(); ringIndex++) {
            std::vector<Slice> level1UserMemSlice;
            for (auto &cclSlice : level0DataSegsSlice[ringIndex]) {
                Slice tmpSlice;
                tmpSlice.size = cclSlice.size;
                CHK_PRT_RET(execMem.outputMem.size() == 0,
                    HCCL_ERROR("[CollReduceScatterRingFor91093Executor][KernelRun]cclout memsize[%llu] is zero",
                    execMem.outputMem.size()), HCCL_E_PARA);              
                tmpSlice.offset =
                    (cclSlice.offset / execMem.outputMem.size()) * param.DataDes.count * perDataSize +
                    multiStreamSlice[ringIndex][0].offset;
                level1UserMemSlice.push_back(tmpSlice);
                HCCL_DEBUG("rank[%u], ringIndex[%u], tmpSlice.offset=[%llu], size=[%llu]",
                    topoAttr_.userRank, ringIndex, tmpSlice.offset, tmpSlice.size);
            }
            multRingsUserMemSlice.push_back(level1UserMemSlice);
        }
    }
    // 区分消减拷贝场景
    if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING &&
        (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB)) {
        // 图模式opinfo不为空
        HcomCollOpInfo graphModeOpInfo = {
            "", execMem.inputMem.ptr(), nullptr, param.DataDes.count, param.DataDes.dataType,
            param.root, param.reduceType};
        CHK_RET(RunIntraSeverReduceScatter(param.tag, execMem.inputMem, execMem.scratchMem, execMem.count,
            param.DataDes.dataType, param.reduceType, level0DataSegsSlice,
            param.stream, PROF_STAGE_1, 0, &graphModeOpInfo, multRingsUserMemSlice, param.retryEnable));
    } else if (opInfoPtr != nullptr && (innerRankSize > 1 || level2RankSize > 1)) {
        HcomCollOpInfo opInfoByReduceScatterDMAreduce = *opInfoPtr;
        opInfoByReduceScatterDMAreduce.outputAddr      = nullptr;
        CHK_RET(RunIntraSeverReduceScatter(param.tag, execMem.inputMem, execMem.scratchMem, execMem.count,
            param.DataDes.dataType, param.reduceType, level0DataSegsSlice,
            param.stream, PROF_STAGE_1, 0, &opInfoByReduceScatterDMAreduce, multRingsUserMemSlice, param.retryEnable));
    } else {
        CHK_RET(RunIntraSeverReduceScatter(param.tag, execMem.inputMem, execMem.scratchMem, execMem.count,
            param.DataDes.dataType, param.reduceType,
            level0DataSegsSlice, param.stream, PROF_STAGE_1, 0, opInfoPtr, multRingsUserMemSlice, param.retryEnable));
    }
    // 对于单server图模式的最后一步需要把数据从ccl input拷贝到ccl output上
    if (innerRankSize == 1 && level2RankSize == 1 && opInfoPtr == nullptr) {
        DeviceMem srcMem = execMem.inputMem.range(topoAttr_.userRank * execMem.outputMem.size(),
            execMem.outputMem.size());
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, execMem.outputMem, srcMem, const_cast<Stream&>(param.stream)));
    }

    if  (innerRankSize > 1) {
        // 节点间做reduce scatter（ring/NHR)
        u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.scratchMem, param.DataDes.dataType, param.reduceType);
        std::unique_ptr<ExecutorBase> innerExecutor;

        // 计算slice
        std::vector<Slice> level1DataSegsSlice;

        CHK_RET(CalLevel1DataSegsSlice(execMem, commIndex, sliceNum, innerRankSize, level2RankSize,
            level1DataSegsSlice));

        if (GetExternalInputEnableRdmaSdmaConcurrent() && (execMem.outputMem.size() >= HCCL_SPLIT_SIZE_INTER_SERVER)
            && !aicpuUnfoldMode_) {
            u32 syncTrans = (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) ? BEST_SPLIT_VALUE_DR :
                BEST_SPLIT_VALUE_SR;
            CHK_RET(Level1ReduceScatterConcurrent(execMem.inputMem, execMem.scratchMem, execMem.count,
                param.DataDes.dataType, param.reduceType, param.stream, PROF_STAGE_2,
                level1DataSegsSlice, syncTrans, reduceAttr));
        } else {
            if (UseInterServerRingAlgo(algType_)) {
                innerExecutor.reset(new (std::nothrow) ReduceScatterRing(dispatcher_, reduceAttr));
                HCCL_INFO("reducescatter ring: using ring algo inter-server.");
            } else if (UseInterServerNBAlgo(algType_)) {
                innerExecutor.reset(new (std::nothrow) ReduceScatterNB(dispatcher_, reduceAttr));
                HCCL_INFO("reducescatter ring: using nonuniform-bruck algo inter-server.");
            } else {
                innerExecutor.reset(new (std::nothrow) ReduceScatterNHR(dispatcher_, reduceAttr));
                HCCL_INFO("reducescatter ring: using nonuniform-hierarchical-ring algo inter-server.");
            }
            CHK_SMART_PTR_NULL(innerExecutor);

            CHK_RET(innerExecutor->Prepare(execMem.inputMem, execMem.inputMem, execMem.scratchMem, execMem.count,
                param.DataDes.dataType, param.stream, param.reduceType, OUTER_BRIDGE_RANK_ID, level1DataSegsSlice));
            CHK_RET(innerExecutor->RegisterProfiler(
                (innerRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + innerCommInfo.localRank,
                PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET, param.stream));
            CHK_RET(RunTemplate(innerExecutor, innerCommInfo));
        }
    }

    if (level2RankSize > 1) {
        /* ****************** 超节点间 reducescatter *******************************/
        u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.scratchMem, param.DataDes.dataType, param.reduceType);
        std::unique_ptr<ExecutorBase> level2Executor;

        // 计算slice
        std::vector<Slice> level2DataSegsSlice;
        for (u32 i = 0; i < level2RankSize; i++) {
            sliceTemp.size = execMem.outputMem.size();
            u32 level2UserRank;
            CHK_RET(GetUserRankByRank(COMM_LEVEL2, COMM_INDEX_0, i, level2UserRank));
            sliceTemp.offset = level2UserRank * execMem.outputMem.size();
            level2DataSegsSlice.push_back(sliceTemp);
            HCCL_DEBUG("rank[%u], level2DataSegsSlice[%u].offset=%llu, size=[%llu], level2RankSize[%u]",
                topoAttr_.userRank, i, sliceTemp.offset, sliceTemp.size, level2RankSize);
        }

        level2Executor.reset(new (std::nothrow) ReduceScatterRing(dispatcher_, reduceAttr));
        HCCL_INFO("reducescatter ring: using ring algo inter-superPod.");

        CHK_SMART_PTR_NULL(level2Executor);

        CHK_RET(level2Executor->Prepare(execMem.inputMem, execMem.inputMem, execMem.scratchMem, execMem.count,
            param.DataDes.dataType, param.stream, param.reduceType, OUTER_BRIDGE_RANK_ID, level2DataSegsSlice));
        CHK_RET(level2Executor->RegisterProfiler(
            (level2RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2CommInfo.localRank,
            PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET, param.stream));
        CHK_RET(RunTemplate(level2Executor, level2CommInfo));
    }

    if (innerRankSize > 1 || level2RankSize > 1) {
        // 区分消减拷贝场景（消减拷贝数据需要拷贝到user output上）
        DeviceMem srcMem = execMem.inputMem.range(topoAttr_.userRank * execMem.outputMem.size(),
            execMem.outputMem.size());
        if (opInfoPtr != nullptr) {
            DeviceMem dstMem = DeviceMem::create(static_cast<u8 *>(opInfoPtr->outputAddr), execMem.outputMem.size());
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream)));
        } else {
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, execMem.outputMem, srcMem, const_cast<Stream&>(param.stream)));
        }
    }

    HCCL_INFO("reducescatter double ring run success");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AlignedReduceScatterAsymDoubleRingExecutor", AlignedReduceScatterAsymDoubleRing,
    CollAlignedReduceScatterAsymDoubleRingExecutor);
}
