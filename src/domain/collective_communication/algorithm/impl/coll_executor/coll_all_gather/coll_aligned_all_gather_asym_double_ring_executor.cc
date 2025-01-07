/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "coll_aligned_all_gather_asym_double_ring_executor.h"
#include "hccl_types.h"

namespace hccl {
CollAlignedAllGatherAsymDoubleRingExecutor::CollAlignedAllGatherAsymDoubleRingExecutor(
    const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAllGatherRingFor91093Executor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE;
}

HcclResult CollAlignedAllGatherAsymDoubleRingExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcCombineCommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAlignedAllGatherAsymDoubleRingExecutor::CalcTransportMemType(TransportMemType &inputType,
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

HcclResult CollAlignedAllGatherAsymDoubleRingExecutor::CalcCombineCommInfo(TransportMemType inputType,
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

HcclResult CollAlignedAllGatherAsymDoubleRingExecutor::DoubleRingAllGather(
    const std::string &tag, DeviceMem inputMem, DeviceMem outputMem,
    const u64 count, const HcclDataType dataType, const std::vector<std::vector<Slice> > multRingsSliceZero,
    Stream stream, s32 profStage, const u64 baseOffset, const HcomCollOpInfo *opInfo,
    const std::vector<std::vector<Slice>> multRingsUserMemSlice)
{
    (void)tag;
    HCCL_INFO("[CollAlignedAllGatherAsymDoubleRingExecutor][DoubleRingAllGather] DoubleRingAllGather starts");
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
    for (int i = 0; i < 32; i++) {
        nicList.push_back(i);
    }

    // std::cout << "nicList: ";
    // for (auto id : nicList) {
    //     std::cout << id << ' ';
    // }
    // std::cout << std::endl;

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
        HCCL_ERROR("[CollAlignedAllGatherAsymDoubleRingExecutor][DoubleRingAllGather]Double ring "
        "all gather failed, return[%d]", ret), ret);
    u32 ringIndexOp = COMM_INDEX_0;
    u32 rankSize = outerZeroCommInfo.localRankSize;
    ret = executor->RegisterProfiler(
        ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
        (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + outerZeroCommInfo.localRank,
        profStage, HCCL_EXEC_STEP_NOT_SET, stream);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAlignedAllGatherAsymDoubleRingExecutor][DoubleRingAllGather]Double ring "
        "all gather failed, return[%d]", ret), ret);

    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    ret = RunTemplate(executor, outerZeroCommInfo);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAlignedAllGatherAsymDoubleRingExecutor][DoubleRingAllGather] Double ring "
                   "reduce scatter failed failed,return[%d]", ret), ret);

    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}

HcclResult CollAlignedAllGatherAsymDoubleRingExecutor::RunIntraSeverAllGather(
    const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
    const u64 count, const HcclDataType &dataType, const std::vector<std::vector<Slice>> &multRingsSliceZero,
    const Stream &stream, s32 profStage, const u64 baseOffset, const HcomCollOpInfo *opInfo,
    const std::vector<std::vector<Slice>> &multRingsUserMemSlice)
{
    CHK_RET(DoubleRingAllGather(tag, inputMem, outputMem, count, dataType,
        multRingsSliceZero, stream, profStage, baseOffset, opInfo, multRingsUserMemSlice));
    return HCCL_SUCCESS;
}

HcclResult CollAlignedAllGatherAsymDoubleRingExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_INFO("[CollAlignedAllGatherAsymDoubleRingExecutor][KernelRun] The AllGatherDoubleRingExecutor starts.");
    CHK_RET(ActiveSlaveStreams(param.stream));
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(param.DataDes.dataType, perDataSize));
    CHK_PRT_RET(perDataSize == 0,
        HCCL_ERROR("[CollAlignedAllGatherAsymDoubleRingExecutor][KernelRun]errNo[0x%016llx] datatype[%s] is invalid",
            HCCL_ERROR_CODE(HCCL_E_PARA), GetDataTypeEnumStr(param.DataDes.dataType).c_str()), HCCL_E_PARA);

    // CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    // SubCommInfo outerCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    // u32 level0ServerIndex = outerCommInfo.localRank;
    // CHK_RET(CheckCommSize(COMM_LEVEL1, level0ServerIndex + 1));
    // SubCommInfo innerCommInfo = GetSubCommInfo(COMM_LEVEL1, level0ServerIndex);
    // CHK_RET(CheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1));
    // SubCommInfo level2CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);
    // 获取打平通信域
    CHK_RET(CheckCommSize(COMM_COMBINE_ORDER, COMM_INDEX_0 + 1));
    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_COMBINE_ORDER, COMM_INDEX_0);

    //  第一步，将数据从input内存拷贝到output内存的对应位置
    // u32 level1ServerIndex = innerCommInfo.localRank;
    u32 level0RankSize = outerCommInfo.localRankSize;
    // u32 level1RankSize = innerCommInfo.localRankSize;
    // u32 level2RankSize = level2CommInfo.localRankSize;
    u32 level1RankSize = 1;
    u32 level2RankSize = 1;

    u64 inputMemSize = execMem.inputMem.size();
    u64 dstMemOffset = topoAttr_.userRank * inputMemSize;
    DeviceMem dstMem = execMem.outputMem.range(dstMemOffset, inputMemSize);
    CHK_SMART_PTR_NULL(dstMem);

    HcomCollOpInfo opInfo = {
        "", execMem.inputPtr, execMem.outputPtr, param.DataDes.count, param.DataDes.dataType, 0, HCCL_REDUCE_RESERVED
    };
    HcomCollOpInfo graphModeOpInfo = {
        "", execMem.inputMem.ptr(), execMem.outputMem.ptr(), param.DataDes.count, param.DataDes.dataType, 0,
        HCCL_REDUCE_RESERVED
    };
    HcomCollOpInfo *opInfoPtr = nullptr;
    if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
        opInfoPtr = &graphModeOpInfo;
    }

    // 图模式opinfo不为空，但需要将数据从ccl input拷贝到ccl output上
    HcclResult ret = HCCL_SUCCESS;
    if (!DMAReduceFlag_) {
        ret = HcclD2DMemcpyAsync(dispatcher_, dstMem, execMem.inputMem, const_cast<Stream&>(param.stream));
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[CollAlignedAllGatherAsymDoubleRingExecutor][KernelRun]all gather double "
                        "ring memcpy Failed, Offset[%llu], Size[%llu]", dstMemOffset, inputMemSize), ret);
    } else {
        opInfoPtr = &opInfo;
        // // 先做server间算法，带有消减拷贝场景数据需要从user input取，拷贝到ccl output上
        // if (level1RankSize > 1 || level2RankSize > 1) {
        //     DeviceMem srcMem = DeviceMem::create(static_cast<u8 *>(execMem.inputPtr), inputMemSize);
        //     ret = HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream));
        //     CHK_PRT_RET(ret != HCCL_SUCCESS,
        //         HCCL_ERROR("[CollAlignedAllGatherAsymDoubleRingExecutor][KernelRun]all gather double "
        //             "ring user memcpy Failed, Offset[%llu], Size[%llu]", dstMemOffset, inputMemSize), ret);
        // }
    }
    // if (level2RankSize > 1) {
    //     std::unique_ptr<ExecutorBase> level2AGExecutor;
    //     level2AGExecutor.reset(new (std::nothrow) AllGatherRing(dispatcher_));
    //     HCCL_INFO("allgather ring: using ring algo inter-server.");
    //     CHK_SMART_PTR_NULL(level2AGExecutor);

    //     std::vector<Slice> level2DataSegsSlice;
    //     for (u32 i = 0; i < level2RankSize; i++) {
    //         Slice sliceTemp;
    //         sliceTemp.size = inputMemSize;
    //         sliceTemp.offset = i * level1RankSize * level0RankSize * inputMemSize +
    //             (level1ServerIndex * level0RankSize + level0ServerIndex) * inputMemSize;
    //         level2DataSegsSlice.push_back(sliceTemp);
    //     }
    //     CHK_RET(level2AGExecutor->Prepare(execMem.outputMem, execMem.outputMem, execMem.inputMem, execMem.count,
    //         param.DataDes.dataType, param.stream,
    //         HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, level2DataSegsSlice, 0));

    //     CHK_RET(level2AGExecutor->RegisterProfiler((
    //         level2RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2CommInfo.localRank,
    //         PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, param.stream));

    //     CHK_RET(RunTemplate(level2AGExecutor, level2CommInfo));
    //     HCCL_INFO("allgather double ring [superpod] level2 allgather run success");
    // }
    // if (level1RankSize > 1) {
    //     // 计算slice, 不同超节点相同slice
    //     std::vector<Slice> level1DataSegsSlice;
    //     for (u32 j = 0; j < level1RankSize; j++) {
    //         for (u32 i = 0; i < level2RankSize; i++) {
    //             Slice level1Slice;
    //             level1Slice.size = inputMemSize;
    //             level1Slice.offset =
    //                 (j * level0RankSize +  i * level1RankSize * level0RankSize + level0ServerIndex) * inputMemSize;
    //             level1DataSegsSlice.push_back(level1Slice);
    //         }
    //     }
        
    //     if (GetExternalInputEnableRdmaSdmaConcurrent() && (inputMemSize >= HCCL_SPLIT_SIZE_INTER_SERVER) 
    //         && !aicpuUnfoldMode_) {
    //         u32 syncTrans = (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) ? BEST_SPLIT_VALUE_DR :
    //             BEST_SPLIT_VALUE_SR;
    //         CHK_RET(Level1AllGatherConcurrent(execMem.inputMem, execMem.outputMem, execMem.count, param.DataDes.dataType,
    //             param.stream, PROF_STAGE_1, level1DataSegsSlice, syncTrans));
    //     } else {
    //         std::unique_ptr<ExecutorBase> level1AGExecutor;
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
    //         CHK_RET(level1AGExecutor->Prepare(execMem.outputMem, execMem.outputMem, execMem.inputMem, execMem.count,
    //             param.DataDes.dataType, param.stream,
    //             HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, level1DataSegsSlice, 0));

    //         CHK_RET(level1AGExecutor->RegisterProfiler((
    //             level1RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2CommInfo.localRank,
    //             PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));

    //         CHK_RET(RunTemplate(level1AGExecutor, innerCommInfo));
    //         HCCL_INFO("allgather double ring [superpod] level1 allgather run success");
    //     }
    // }
    // 节点内做all gather double ring
    std::vector<Slice> dataSegsSlice;
    std::vector<std::vector<Slice>> multRingsSliceZero; // 数据基于该rank上环0的偏移
    CHK_RET(PrepareAllgatherSlice(level0RankSize, inputMemSize, dataSegsSlice));

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
    std::vector<std::vector<Slice>> multRingsSlice;
    for (u32 ringIndex = 0; ringIndex < multRingsSliceZero.size(); ringIndex++) {
        std::vector<Slice> level2DataSlice;
        CHK_RET(CalculateLevel2AllgatherSlice(inputMemSize, level0RankSize, level1RankSize, level2RankSize,
            multRingsSliceZero, level2DataSlice, ringIndex));
        multRingsSlice.push_back(level2DataSlice);
    }

    std::vector<std::vector<Slice>> multRingsUserMemSlice;
    if (!DMAReduceFlag_) {
        multRingsUserMemSlice = multRingsSlice;
    } else {
        for (u32 ringIndex = 0; ringIndex < multRingsSlice.size(); ringIndex++) {
            std::vector<Slice> userMemSlice;
            for (auto &cclSlice : multRingsSlice[ringIndex]) {
                Slice tmpSlice;
                tmpSlice.size = cclSlice.size;
                tmpSlice.offset =
                    (cclSlice.offset / inputMemSize) * opInfo.count* perDataSize +
                    multRingsSliceZero[ringIndex][0].offset;
                userMemSlice.push_back(tmpSlice);
                HCCL_DEBUG("rank[%u], ringIndex[%u], tmpSlice.offset=[%llu], size=[%llu]",
                    topoAttr_.userRank, ringIndex, tmpSlice.offset, tmpSlice.size);
            }
            multRingsUserMemSlice.push_back(userMemSlice);
        }
    }
    if (DMAReduceFlag_ && (level1RankSize > 1 || level2RankSize > 1)) {
        // allgather输入放在CCL buffer上，通过设置nullptr指示要从CCL buffer获取输入
        opInfo.inputAddr = nullptr;
    }
    CHK_RET(RunIntraSeverAllGather(param.tag, execMem.inputMem, execMem.outputMem, execMem.count,
        param.DataDes.dataType, multRingsSlice, param.stream, PROF_STAGE_2, 0, opInfoPtr, multRingsUserMemSlice));
    HCCL_INFO("allgather double ring run success");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AlignedAllGatherAsymDoubleRingExecutor", AlignedAllGatherAsymDoubleRing,
    CollAlignedAllGatherAsymDoubleRingExecutor);

} // namespace hccl
