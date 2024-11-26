/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BROADCAST_ONESHOT_PUB_H
#define BROADCAST_ONESHOT_PUB_H

#include "executor_base_pub.h"

namespace hccl {
class BroadcastHD : public ExecutorBase {
public:
    explicit BroadcastHD(const HcclDispatcher dispatcher, std::vector<Stream> &meshStreams,
        const std::vector<std::shared_ptr<LocalNotify>> &meshSignal,
        const std::vector<std::shared_ptr<LocalNotify>> &meshSignalAux, u32 interRank, const HcomCollOpInfo *opInfo);
    ~BroadcastHD() override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links) override;

protected:
private:
    HcclResult MainRecordSub();
    HcclResult SubWaitMain();
    HcclResult MainWaitSub();
    HcclResult SubRecordMain();
    HcclResult PrepareStep(u32 rankSize);
    HcclResult RunFinalStep(u32 rank, u32 rankSize, const std::vector<LINK> &links);
    HcclResult RunSend(u32 rank, u32 step, u32 rankSize, const std::vector<LINK> &links);
    HcclResult RunSendFirst(u32 rank, u32 rankSize, const std::vector<LINK> &links);
    HcclResult RunReceive(u32 rank, u32 step, u32 rankSize, const std::vector<LINK> &links);
    HcclResult RunReceiveFirst(u32 rank, u32 rankSize, const std::vector<LINK> &links);
    u32 GetDstRank(u32 rank, u32 step, u32 rankSize);
    u32 localRank_;
    std::vector<Stream> meshStreams_;                                /* * 多steam* */
    const std::vector<std::shared_ptr<LocalNotify>> &meshSignal_;    /* 每个ring创建一个signal */
    const std::vector<std::shared_ptr<LocalNotify>> &meshSignalAux_; /* 从stream wait，主steam record */
    const HcomCollOpInfo *opInfo_;
    DeviceMem emptyMem_;
    std::map<u32, u32> stepMap_;
    u32 nSteps_ = 0;
    const u32 base = 2;
};
}  // namespace hccl
#endif /* BROADCAST_ONESHOT_PUB_H */