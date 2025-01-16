/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_aligned_all_reduce_asym_double_ring_executor.h"

namespace hccl {

CollAlignedAllReduceAsymDoubleRingExecutor::CollAlignedAllReduceAsymDoubleRingExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAllReduceRingFor91093Executor(dispatcher, topoMatcher)
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        DMAReduceFlag_ = true;
    } else {
        DMAReduceFlag_ = false;
    }
}

HcclResult CollAlignedAllReduceAsymDoubleRingExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcCombineCommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAlignedAllReduceAsymDoubleRingExecutor::CalcTransportMemType(TransportMemType &inputType,
    TransportMemType &outputType)
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::PARAM_OUTPUT;
    }
    HCCL_INFO("[CollAllGatherRingFor91093Executor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAlignedAllReduceAsymDoubleRingExecutor::CalcCombineCommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commCombinePara(COMM_COMBINE_ORDER, CommType::COMM_TAG_MESH);
    CHK_RET(CalcCommPlaneInfo(tag_, commCombinePara, opTransport[COMM_COMBINE_ORDER], inputType, outputType));

    LevelNSubCommTransport &commTransportLevel0 = opTransport[COMM_COMBINE_ORDER];
    for (u32 subCommIndex = 0; subCommIndex < commTransportLevel0.size(); subCommIndex++) {
        for (auto &transportRequest : commTransportLevel0[subCommIndex].transportRequests) {
            transportRequest.isUsedRdma = topoAttr_.isUsedRdmaMap.at(transportRequest.remoteUserRank);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CollAlignedAllReduceAsymDoubleRingExecutor::DoubleRingReduceScatter(const std::string &tag,
    DeviceMem inputMem, DeviceMem outputMem, const u64 count, const HcclDataType dataType,
    const HcclReduceOp reductionOp, const std::vector<std::vector<Slice>> multRingsSliceZero, Stream stream,
    s32 profStage, const u64 baseOffset, const HcomCollOpInfo *opInfo,
    const std::vector<std::vector<Slice>> multRingsUserMemSlice, const bool retryEnable)
{
    (void)tag;
    HCCL_INFO(
        "[CollAlignedAllReduceAsymDoubleRingExecutor][DoubleRingReduceScatter] DoubleRingReduceScatter starts");
    HcclResult ret = HCCL_SUCCESS;
    u32 ringNum = multRingsSliceZero.size();
    // CHK_RET(CheckCommSize(COMM_LEVEL0, ringNum));

    // // 拿到ring环映射关系
    // SubCommInfo outerZeroCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    // auto nicList = topoAttr_.nicList;

    CHK_RET(CheckCommSize(COMM_COMBINE_ORDER, COMM_INDEX_0 + 1));
    SubCommInfo outerZeroCommInfo = GetSubCommInfo(COMM_COMBINE_ORDER, COMM_INDEX_0);
    // auto nicList = topoAttr_.nicList;
    std::vector<u32> nicList;
    for (int i = 0; i < outerZeroCommInfo.localRankSize; i++) {
        nicList.push_back(i);
    }
    HCCL_INFO("nicList reset by outerZeroCommInfo.localRankSize");

    std::vector<std::vector<u32>> multiRingsOrder =
        GetRingsOrderByTopoType(outerZeroCommInfo.localRankSize, topoType_, nicList);

    u64 reduceAttr = GetReduceAttr(inputMem, outputMem, dataType, reductionOp);

    // SubCommInfo outerRingCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    SubCommInfo outerRingCommInfo = GetSubCommInfo(COMM_COMBINE_ORDER, COMM_INDEX_0);
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
    executor.reset(new (std::nothrow) AlignedReduceScatterAsymDoubleRing(
        dispatcher_, reduceAttr, opInfo, topoAttr_.userRank, algResResp_->slaveStreams,
        algResResp_->notifiesM2S, algResResp_->notifiesS2M, rankOrders, userMemInputSlicesOfDoubleRing));
    CHK_SMART_PTR_NULL(executor);
    ret = executor->Prepare(inputMem, inputMem, outputMem, count, dataType, stream, multRingsSliceZero,
        reductionOp, OUTER_BRIDGE_RANK_ID, baseOffset, retryEnable);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAlignedAllReduceAsymDoubleRingExecutor][DoubleRingReduceScatter] Double ring reduce scatter failed"
        "failed,return[%d]", ret), ret);
    u32 ringIndexOp = COMM_INDEX_0;
    u32 rankSize = outerRingCommInfo.localRankSize;
    ret = executor->RegisterProfiler(
        ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
        (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + outerRingCommInfo.localRank,
        profStage, HCCL_EXEC_STEP_NOT_SET, stream);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAlignedAllReduceAsymDoubleRingExecutor][DoubleRingReduceScatter] Double ring reduce scatter failed "
        "failed,return[%d]", ret), ret);

    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    ret = RunTemplate(executor, outerRingCommInfo);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAlignedAllReduceAsymDoubleRingExecutor][DoubleRingReduceScatter] Double ring reduce scatter failed "
        "failed,return[%d]", ret), ret);

    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}

HcclResult CollAlignedAllReduceAsymDoubleRingExecutor::DoubleRingAllGather(
    const std::string &tag, DeviceMem inputMem, DeviceMem outputMem,
    const u64 count, const HcclDataType dataType, const std::vector<std::vector<Slice> > multRingsSliceZero,
    Stream stream, s32 profStage, const u64 baseOffset, const HcomCollOpInfo *opInfo,
    const std::vector<std::vector<Slice>> multRingsUserMemSlice)
{
    (void)tag;
    HCCL_INFO("[CollAlignedAllReduceAsymDoubleRingExecutor][DoubleRingAllGather] DoubleRingAllGather starts");
    HcclResult ret = HCCL_SUCCESS;
    u32 ringNum = multRingsSliceZero.size();
    // CHK_RET(CheckCommSize(COMM_LEVEL0, ringNum));
    // // 拿到ring环映射关系
    // SubCommInfo outerZeroCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    // 获取打平通信域
    CHK_RET(CheckCommSize(COMM_COMBINE_ORDER, COMM_INDEX_0 + 1));
    SubCommInfo outerZeroCommInfo = GetSubCommInfo(COMM_COMBINE_ORDER, COMM_INDEX_0);
    // auto nicList = topoAttr_.nicList;
    std::vector<u32> nicList;
    for (int i = 0; i < outerZeroCommInfo.localRankSize; i++) {
        nicList.push_back(i);
    }
    std::vector<std::vector<u32>> multiRingsOrder =
        GetRingsOrderByTopoType(outerZeroCommInfo.localRankSize, topoType_, nicList);
    // 生成两个ring上的userMemOut_上对应的slices
    std::vector<std::vector<Slice>> userMemOutputSlicesOfDoubleRing;
    CHK_RET(CollectMultiRingsUserMemSlices(ringNum, dataType, opInfo, multRingsSliceZero,
        multiRingsOrder, multRingsUserMemSlice, userMemOutputSlicesOfDoubleRing));
    // 生成两个ring上的rankOrder
    std::vector<std::vector<u32>> rankOrders;
    CHK_RET(CollectMultiRingsRankOrder(ringNum, multiRingsOrder, rankOrders));
    // 初始化executor
    std::unique_ptr<ExecutorBase> executor;
    executor.reset(new (std::nothrow) AlignedAllGatherAsymDoubleRing(dispatcher_,
        opInfo, topoAttr_.userRank, algResResp_->slaveStreams, algResResp_->notifiesM2S,
        algResResp_->notifiesS2M, rankOrders, userMemOutputSlicesOfDoubleRing));
    CHK_SMART_PTR_NULL(executor);

    ret = executor->Prepare(outputMem, outputMem, inputMem, count, dataType, stream, multRingsSliceZero,
        HCCL_REDUCE_RESERVED, OUTER_BRIDGE_RANK_ID, baseOffset);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAlignedAllReduceAsymDoubleRingExecutor][DoubleRingAllGather]Double ring "
        "all gather failed, return[%d]", ret), ret);
    u32 ringIndexOp = COMM_INDEX_0;
    u32 rankSize = outerZeroCommInfo.localRankSize;
    ret = executor->RegisterProfiler(
        ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
        (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + outerZeroCommInfo.localRank,
        profStage, HCCL_EXEC_STEP_NOT_SET, stream);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAlignedAllReduceAsymDoubleRingExecutor][DoubleRingAllGather]Double ring "
        "all gather failed, return[%d]", ret), ret);

    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    ret = RunTemplate(executor, outerZeroCommInfo);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAlignedAllReduceAsymDoubleRingExecutor][DoubleRingAllGather] Double ring "
                   "reduce scatter failed failed,return[%d]", ret), ret);

    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}

HcclResult CollAlignedAllReduceAsymDoubleRingExecutor::RunIntraSeverReduceScatter(
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

HcclResult CollAlignedAllReduceAsymDoubleRingExecutor::RunIntraSeverAllGather(
    const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
    const u64 count, const HcclDataType &dataType, const std::vector<std::vector<Slice>> &multRingsSliceZero,
    const Stream &stream, s32 profStage, const u64 baseOffset, const HcomCollOpInfo *opInfo,
    const std::vector<std::vector<Slice>> &multRingsUserMemSlice)
{
    CHK_RET(DoubleRingAllGather(tag, inputMem, outputMem, count, dataType,
        multRingsSliceZero, stream, profStage, baseOffset, opInfo, multRingsUserMemSlice));
    return HCCL_SUCCESS;
}

HcclResult CollAlignedAllReduceAsymDoubleRingExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_INFO("[CollAlignedAllReduceAsymDoubleRingExecutor][Run]The CollAlignedAllReduceAsymDoubleRingExecutor starts");
    
    CHK_RET(ActiveSlaveStreams(param.stream));
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(param.DataDes.dataType, perDataSize));
    std::vector<Slice> dataSegsSlice; // 数据分成ranksize份，每份的起始偏移和大小
    std::vector<std::vector<Slice> > multRingsSliceZero; // 数据基于该rank上环0的偏移
    // CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    // SubCommInfo outerCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    // 获取打平通信域
    CHK_RET(CheckCommSize(COMM_COMBINE_ORDER, COMM_INDEX_0 + 1));
    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_COMBINE_ORDER, COMM_INDEX_0);

    u32 sliceNum = outerCommInfo.localRankSize;

    // 根据数据量计算每个环上数据的偏移和大小
    CHK_RET(ExecutorBase::PrepareSliceData(execMem.count, perDataSize, sliceNum, 0, dataSegsSlice));

    // /* 三步算法step1：外层 - 节点内 reduce-scatter */
    // // 构造ring algorithm对应的reduce-scatter实例

    //  多环数据切分
    if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
        // multRingsSliceZero = PrepareMultiRingSlice(dataSegsSlice, param.tag, false, topoAttr_.nicList);
        // 双环数据相同
        for (int i = 0; i < 2; ++i) {
            multRingsSliceZero.push_back(dataSegsSlice);
        }    
        // 获取第二个环上的数据
        std::vector<Slice>& secondVector = multRingsSliceZero[1];
        // 调整第二个环上的Slice顺序，按照 [0, 7, 6, 5, 4, 3, 2, 1] 的顺序进行调整
        size_t n = secondVector.size();
        for (size_t i = 1; i < n / 2; ++i) {
            std::swap(secondVector[i], secondVector[n - i]);
        }
    } else {
        multRingsSliceZero.push_back(dataSegsSlice);
    }

    u32 ringNum;
    if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
        ringNum = OUTER_PLANE_NUM_IN_NPRING_DOUBLE;
    } else {
        ringNum = OUTER_PLANE_NUM_IN_NPRING_SINGLE;
    }

    // multRingsSliceZero = ASYMMultiRingSlicePrepare(ringNum, sliceNum, false, execMem.outputMem,
    //     dataSegsSlice, param.tag);

    // printf("multRingsSliceZero.size(): %d\n", multRingsSliceZero.size());
    // printf("multRingsSliceZero[0].size(): %d\n", multRingsSliceZero[0].size());
    // for (size_t i = 0; i < multRingsSliceZero.size(); ++i) {
    //     std::cout << "Ring " << i << ":\n";
    //     for (size_t j = 0; j < multRingsSliceZero[i].size(); ++j) {
    //         const Slice& slice = multRingsSliceZero[i][j];
    //         std::cout << "  Slice " << j << " - Offset: " << slice.offset << ", Size: " << slice.size << " bytes\n";
    //     }
    // }

    // 第一步的reducescatter输出放在CCL buffer上，通过设置nullptr指示不做最后一步的DMA削减动作
    HcomCollOpInfo reduceScatterOpInfo = {
        "", execMem.inputPtr, nullptr, execMem.count, param.DataDes.dataType, param.root, param.reduceType
    };
    // HcomCollOpInfo reduceScatterOpInfo = {
    //     "", execMem.inputPtr, execMem.outputPtr, execMem.count, param.DataDes.dataType, param.root, param.reduceType
    // };
    HcomCollOpInfo reduceScatterGraphModeOpInfo = {
        "", execMem.inputMem.ptr(), nullptr, execMem.count, param.DataDes.dataType, param.root, param.reduceType
    };
    HcomCollOpInfo *reduceScatterOpInfoPtr = nullptr;
    if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
        reduceScatterOpInfoPtr = &reduceScatterGraphModeOpInfo;
    }
    if (DMAReduceFlag_) {
        reduceScatterOpInfoPtr = &reduceScatterOpInfo;
    }
    const std::vector<std::vector<Slice>> multRingsUserMemSliceDefault = std::vector<std::vector<Slice>>(0);
    CHK_RET(RunIntraSeverReduceScatter(param.tag, execMem.inputMem, execMem.outputMem, execMem.count,
        param.DataDes.dataType, param.reduceType, multRingsSliceZero, param.stream,
        PROF_STAGE_0, 0, reduceScatterOpInfoPtr, multRingsUserMemSliceDefault, param.retryEnable));
    HCCL_INFO("allreduce double ring stage0 run success.");

    bool isSelectAHC = (UseInterServerAHCAlgo(algType_) || UseInterServerAHCBrokeAlgo(algType_));

    for (size_t i = 0; i < outerCommInfo.localRankSize; ++i) {
        DeviceMem src = execMem.inputMem;
        DeviceMem dst = DeviceMem::create(execMem.outputPtr,
                execMem.inputMem.size);
        HcclD2DMemcpyAsync(dispatcher_, dst, src, const_cast<Stream&>(param.stream));
    }

    // /* 三步算法step2: 内层 - 节点间 allreduce */
    // u64 hdSize;
    // u32 segmentIdx;
    // u32 commIndex;
    // CHK_RET(PrepareInnerCommInfo(segmentIdx, commIndex, hdSize, outerCommInfo, multRingsSliceZero, param.tag));

    // u64 hdCount = hdSize / perDataSize;
    // if (topoAttr_.superPodNum <= 1 || isSelectAHC) {
    //     bool isConcurrent = (topoAttr_.moduleNum > 1) && !isSelectAHC && !aicpuUnfoldMode_;
    //     if (GetExternalInputEnableRdmaSdmaConcurrent() && (hdSize >= HCCL_SPLIT_SIZE_INTER_SERVER) && isConcurrent) {
    //         u32 syncTrans = (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) ? BEST_SPLIT_VALUE_DR :
    //         BEST_SPLIT_VALUE_SR;
    //         CHK_RET(Level1AllReduceConcurrent(execMem.inputMem, execMem.outputMem, execMem.count,
    //             param.DataDes.dataType, param.reduceType, param.stream, PROF_STAGE_1,
    //             dataSegsSlice, segmentIdx, commIndex, hdSize, syncTrans));
    //     } else {
    //         DeviceMem allreduceInput = execMem.inputMem.range(dataSegsSlice[segmentIdx].offset, hdSize);
    //         CHK_SMART_PTR_NULL(allreduceInput);
    //         DeviceMem allreduceOutput = execMem.outputMem.range(dataSegsSlice[segmentIdx].offset, hdSize);
    //         CHK_SMART_PTR_NULL(allreduceOutput);

    //         CommPlane commPlaneLevel1 = isSelectAHC ? COMM_LEVEL1_AHC : COMM_LEVEL1;
    //         CHK_RET(CheckCommSize(commPlaneLevel1, commIndex + 1));
    //         SubCommInfo innerCommInfo = GetSubCommInfo(commPlaneLevel1, commIndex);

    //         u64 reduceAttr = GetReduceAttr(allreduceInput, allreduceOutput, param.DataDes.dataType, param.reduceType);
    //         std::unique_ptr<ExecutorBase> innerExecutor;
    //         if (UseInterServerRingAlgo(algType_)) {
    //             innerExecutor.reset(new (std::nothrow) AllReduceRing(dispatcher_, reduceAttr));
    //             HCCL_INFO("allreduce ring: using ring algo inter-server.");
    //         } else if (UseInterServerNHRV1Algo(algType_)) {
    //             innerExecutor.reset(new (std::nothrow) AllReduceNHRV1(dispatcher_, reduceAttr));
    //             HCCL_INFO("allreduce ring: using nhr_v1 algo inter-server.");
    //         } else if (UseInterServerAHCAlgo(algType_)) {
    //             // 获取通信域分组信息
    //             std::vector<std::vector<u32>> subGroups;
    //             CHK_RET(topoMatcher_->GetLevelSubGroups(commPlaneLevel1, subGroups));
    //             innerExecutor.reset(new (std::nothrow) AllReduceAHC(dispatcher_, reduceAttr, execMem.count, subGroups));
    //             HCCL_INFO("allreduce ring: using ahc algo inter-server.");
    //         } else if (UseInterServerAHCBrokeAlgo(algType_)) {
    //             // 获取通信域分组信息
    //             std::vector<std::vector<u32>> subGroups;
    //             CHK_RET(topoMatcher_->GetLevelSubGroups(commPlaneLevel1, subGroups));
    //             innerExecutor.reset(new (std::nothrow) AllReduceAHCBroke(dispatcher_, reduceAttr, execMem.count, subGroups));
    //             HCCL_INFO("allreduce ring: using ahc-broke algo inter-server.");
    //         } else if (UseInterServerNBAlgo(algType_)) {
    //             innerExecutor.reset(new (std::nothrow) AllReduceNB(dispatcher_, reduceAttr));
    //             HCCL_INFO("allreduce ring: using nonuniform-bruck algo inter-server.");
    //         } else if (UseInterServerNHRAlgo(algType_)) {
    //             u64 curSize = execMem.count * SIZE_TABLE[param.DataDes.dataType]; // 单位 byte
    //             HCCL_DEBUG("allreduce ring: curSize[%llu] deviceNumPerAggregation[%u] commOuterSize[%u]",
    //                 curSize, topoAttr_.deviceNumPerAggregation, outerCommInfo.localRankSize);
    //             if (curSize / topoAttr_.deviceNumPerAggregation <= NHR_ALLREDUCE_SMALL_SIZE) {
    //                 innerExecutor.reset(new (std::nothrow) AllReduceNHROneshot(dispatcher_, reduceAttr));
    //             } else {
    //                 innerExecutor.reset(new (std::nothrow) AllReduceNHR(dispatcher_, reduceAttr));
    //             }
    //             HCCL_INFO("allreduce ring: using nhr algo inter-server.");
    //         } else {
    //             HCCL_ERROR("allreduce ring: algType[%u] is not supported.", algType_);
    //             return HCCL_E_NOT_SUPPORT;
    //         }
    //         CHK_SMART_PTR_NULL(innerExecutor);
    //         u32 rankSize = innerCommInfo.localRankSize;
    //         // 节点间的hd 使用环0来记录
    //         CHK_RET(innerExecutor->Prepare(
    //             allreduceInput, allreduceOutput, allreduceOutput, hdCount,
    //             param.DataDes.dataType, param.stream, param.reduceType, OUTER_BRIDGE_RANK_ID,
    //             std::vector<Slice>(0), dataSegsSlice[segmentIdx].offset));
    //         CHK_RET(innerExecutor->RegisterProfiler(
    //             (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + innerCommInfo.localRank,
    //             PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));
    //         CHK_RET(RunTemplate(innerExecutor, innerCommInfo));

    //         HCCL_INFO("allreduce double ring stage1 run success");
    //     }
    // } else {
    //     // 超节点内做reducescatter
    //     CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    //     SubCommInfo innerCommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);
    //     u32 level1RankSize = innerCommInfo.localRankSize;
    //     u64 level1Offset = dataSegsSlice[segmentIdx].offset;

    //     // 根据数据量计算每个环上数据的偏移和大小
    //     CHK_RET(ExecutorBase::PrepareSliceData(hdCount, perDataSize, level1RankSize, 0, dataSegsSlice));
    //     DeviceMem reducescatterInput = execMem.inputMem.range(level1Offset, hdSize);
    //     CHK_SMART_PTR_NULL(reducescatterInput);
    //     DeviceMem reducescatterOutput = execMem.outputMem.range(level1Offset, hdSize);
    //     CHK_SMART_PTR_NULL(reducescatterOutput);
    //     if (level1RankSize > 1) {
    //         u64 reduceAttr = GetReduceAttr(reducescatterInput, reducescatterOutput,
    //             param.DataDes.dataType, param.reduceType);
    //         std::unique_ptr<ExecutorBase> level1RSExecutor;

    //         if (UseInterServerRingAlgo(algType_)) {
    //             level1RSExecutor.reset(new (std::nothrow) ReduceScatterRing(dispatcher_, reduceAttr));
    //             HCCL_INFO("reducescatter ring: using ring algo inter-server.");
    //         } else if (UseInterServerNBAlgo(algType_)) {
    //             level1RSExecutor.reset(new (std::nothrow) ReduceScatterNB(dispatcher_, reduceAttr));
    //             HCCL_INFO("reducescatter ring: using nonuniform-bruck algo inter-server.");
    //         } else if (UseInterServerNHRAlgo(algType_)) {
    //             level1RSExecutor.reset(new (std::nothrow) ReduceScatterNHR(dispatcher_, reduceAttr));
    //             HCCL_INFO("reducescatter ring: using nonuniform-hierarchical-ring algo inter-server.");
    //         } else {
    //             HCCL_ERROR("reducescatter ring: algType[%u] is not supported.", algType_);
    //             return HCCL_E_NOT_SUPPORT;
    //         }
    //         CHK_SMART_PTR_NULL(level1RSExecutor);
    //         CHK_RET(level1RSExecutor->Prepare(
    //             reducescatterInput, reducescatterInput, reducescatterOutput, hdCount, param.DataDes.dataType,
    //             param.stream, param.reduceType, OUTER_BRIDGE_RANK_ID, dataSegsSlice, level1Offset));

    //         CHK_RET(level1RSExecutor->RegisterProfiler(
    //             (level1RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + innerCommInfo.localRank,
    //             PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));
    //         CHK_RET(RunTemplate(level1RSExecutor, innerCommInfo));
    //         HCCL_INFO("allreduce double ring [superpod] level1 reducescatter run success");
    //     }

    //     // 超节点间做allreduce
    //     SubCommInfo level2CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);
    //     u32 rankSize = level2CommInfo.localRankSize;
    //     u32 localRank = innerCommInfo.localRank;

    //     DeviceMem allreduceInput =
    //         reducescatterInput.range(dataSegsSlice[localRank].offset, dataSegsSlice[localRank].size);
    //     CHK_SMART_PTR_NULL(allreduceInput);
    //     DeviceMem allreduceOutput =
    //         reducescatterOutput.range(dataSegsSlice[localRank].offset, dataSegsSlice[localRank].size);
    //     CHK_SMART_PTR_NULL(allreduceOutput);

    //     u64 reduceAttr = GetReduceAttr(allreduceInput, allreduceOutput, param.DataDes.dataType, param.reduceType);

    //     std::unique_ptr<ExecutorBase> level2ARExecutor;
    //     if (UseLevel2RingAlgo(algType_)) {
    //         level2ARExecutor.reset(new (std::nothrow) AllReduceRing(dispatcher_, reduceAttr));
    //         HCCL_INFO("allreduce ring: using ring algo level2-server.");
    //     } else {
    //         level2ARExecutor.reset(new (std::nothrow) AllReduceRecursiveHalvingDoubling(dispatcher_, reduceAttr));
    //         HCCL_INFO("allreduce ring: using halving-doubling algo level2-server.");
    //     }
    //     CHK_SMART_PTR_NULL(level2ARExecutor);
    //     u64 arCount = dataSegsSlice[localRank].size / perDataSize;
    //     CHK_RET(level2ARExecutor->Prepare(
    //         allreduceInput, allreduceOutput, allreduceOutput, arCount,
    //         param.DataDes.dataType, param.stream, param.reduceType, OUTER_BRIDGE_RANK_ID,
    //         std::vector<Slice>(0), dataSegsSlice[localRank].offset + level1Offset));
    //     CHK_RET(level2ARExecutor->RegisterProfiler(
    //         (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2CommInfo.localRank,
    //         PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));
    //     CHK_RET(RunTemplate(level2ARExecutor, level2CommInfo));
    //     HCCL_INFO("allreduce double ring [superpod] level2 allreduce run success");

    //     // 超节点内做allgather
    //     if (level1RankSize > 1) {
    //         std::unique_ptr<ExecutorBase> level1AGExecutor;
    //         DeviceMem allgatherInput = execMem.outputMem.range(level1Offset, hdSize);
    //         DeviceMem allgatherOutput = execMem.outputMem.range(level1Offset, hdSize);
    //         if (UseInterServerRingAlgo(algType_)) {
    //             level1AGExecutor.reset(new (std::nothrow) AllGatherRing(dispatcher_));
    //             HCCL_INFO("allgather ring: using ring algo inter-server.");
    //         } else if (UseInterServerNBAlgo(algType_)) {
    //             level1AGExecutor.reset(new (std::nothrow) AllGatherNB(dispatcher_));
    //             HCCL_INFO("allgather ring: using nonuniform-bruck algo inter-server.");
    //         } else if (UseInterServerNHRAlgo(algType_)) {
    //             level1AGExecutor.reset(new (std::nothrow) AllGatherNHR(dispatcher_));
    //             HCCL_INFO("allgather ring: using nonuniform-hierarchical-ring algo inter-server.");
    //         } else {
    //             HCCL_ERROR("allgather ring: algType[%u] is not supported.", algType_);
    //             return HCCL_E_NOT_SUPPORT;
    //         }
    //         CHK_SMART_PTR_NULL(level1AGExecutor);
    //         CHK_RET(level1AGExecutor->Prepare(allgatherInput, allgatherOutput, allgatherOutput, arCount,
    //             param.DataDes.dataType, param.stream,
    //             HcclReduceOp::HCCL_REDUCE_RESERVED, OUTER_BRIDGE_RANK_ID, dataSegsSlice, level1Offset));
    //         CHK_RET(level1AGExecutor->RegisterProfiler(
    //             (level1RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + innerCommInfo.localRank,
    //             PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));
    //         CHK_RET(RunTemplate(level1AGExecutor, innerCommInfo));
    //         HCCL_INFO("allreduce double ring [superpod] level1 allgather run success");
    //     }
    // }
    /* 三步算法step3：外层 - 节点内 allgather */
    // 第三步的allgather输入放在CCL buffer上，通过设置nullptr指示要从CCL buffer获取输入
    HcomCollOpInfo allgatherOpInfo = {
        "", nullptr, execMem.outputPtr, execMem.count, param.DataDes.dataType, param.root, param.reduceType
    };
    HcomCollOpInfo allgatherOpInfoGraphModeOpInfo = {
        "", nullptr, execMem.outputMem.ptr(), execMem.count, param.DataDes.dataType, param.root, param.reduceType
    };
    HcomCollOpInfo *allgatherOpInfoPtr = nullptr;
    if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
        allgatherOpInfoPtr = &allgatherOpInfoGraphModeOpInfo;
    }
    if (DMAReduceFlag_) {
        allgatherOpInfoPtr = &allgatherOpInfo;
    }
    // CHK_RET(RunIntraSeverAllGather(param.tag, execMem.inputMem, execMem.outputMem, hdCount,
    //     param.DataDes.dataType, multRingsSliceZero, param.stream,
    //     PROF_STAGE_2, 0, allgatherOpInfoPtr));
    // CHK_RET(RunIntraSeverAllGather(param.tag, execMem.inputMem, execMem.outputMem, execMem.count,
    //     param.DataDes.dataType, multRingsSliceZero, param.stream,
    //     PROF_STAGE_2, 0, allgatherOpInfoPtr));
    // HCCL_INFO("allreduce double ring stage2 run success");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AlignedAllReduceAsymDoubleRingExecutor", AlignedAllReduceAsymDoubleRing,
    CollAlignedAllReduceAsymDoubleRingExecutor);

}  // namespace hccl
