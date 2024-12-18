/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_ALIGNED_REDUCESCATTER_ASYM_DOUBLE_RING_EXECUTOR_H
#define COLL_ALIGNED_REDUCESCATTER_ASYM_DOUBLE_RING_EXECUTOR_H
#include "coll_reduce_scatter_ring_for_910_93_executor.h"

namespace hccl {
class CollAlignedReduceScatterAsymDoubleRingExecutor : public CollReduceScatterRingFor91093Executor {
public:
    explicit CollAlignedReduceScatterAsymDoubleRingExecutor(
        const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher);
    ~CollAlignedReduceScatterAsymDoubleRingExecutor() = default;

private:
    /* *************** 算法编排 *************** */
    virtual HcclResult DoubleRingReduceScatter(
        const std::string &tag, DeviceMem inputMem, DeviceMem outputMem, const u64 count,
        const HcclDataType dataType, const HcclReduceOp reductionOp,
        const std::vector<std::vector<Slice>> multRingsSliceZero, Stream stream, s32 profStage,
        const u64 baseOffset = 0, const HcomCollOpInfo *opInfo = nullptr,
        const std::vector<std::vector<Slice>> multRingsUserMemSlice = std::vector<std::vector<Slice>>(0),
        const bool retryEnable = false);
    virtual HcclResult RunIntraSeverReduceScatter(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
        const u64 count, const HcclDataType &dataType, const HcclReduceOp &reductionOp,
        const std::vector<std::vector<Slice>> &multRingsSliceZero, const Stream &stream, s32 profStage,
        const u64 baseOffset = 0, const HcomCollOpInfo *opInfo = nullptr,
        const std::vector<std::vector<Slice>> &multRingsUserMemSlice = std::vector<std::vector<Slice>>(0),
        const bool retryEnable = false) override;
    virtual HcclResult KernelRun(const OpParam &param, ExecMem &execMem) override;
    virtual HcclResult CalLevel1DataSegsSlice(const ExecMem &execMem, const u32 &commIndex,
        u32 sliceNum, u32 innerRankSize, u32 level2RankSize,
        std::vector<Slice> &level1DataSegsSlice) override;
};

}  // namespace hccl

#endif
