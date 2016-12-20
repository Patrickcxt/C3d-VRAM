require 'dp'
require 'rnn'
require 'DetReward'
require 'DetLossCriterion'
require 'ThumosDataSet'
require 'MyRecurrentAttention'
require 'nngraph'
require 'MyConstant'
require 'utils'
require 'cutorch'
require 'cudnn'
require 'gnuplot'

local cjson = require 'cjson'

function get_negative_segments(gt_segments)
    local start = gt_segments[{{}, {1}}]
    local val, idx = torch.sort(start, 1)
    local segs = {}
    local pos = 0
    for i = 1, idx:size(1) do
        local st, ed = gt_segments[idx[i][1]][1], gt_segments[idx[i][1]][2]
        table.insert(segs, {pos, st})
        pos = ed
    end
    table.insert(segs, {pos, 1.0})
    return torch.Tensor(segs)
end


function static_idts_density()
    local dh = ThumosDataSet()
    local f = hdf5.open('/home/amax/cxt/data/THUMOS2014/test/idt_density.h5', 'w')
    local iter = 1
    while true do
        print("Iter: ", iter)
        iter = iter + 1
        local video_name, infos, video, labels, segments = dh:getValSample()

        if video_name == nil then
            break
        end

        local num_segments = video:size(1)
        local num_frames = infos[3]
        for j = 1, segments:size(1) do
            local f1, f2 = torch.round(segments[j][1] * num_segments), torch.round(segments[j][2] * num_segments)
            f1, f2 = math.max(1, f1), math.max(1, f2)
            f1, f2 = math.min(f1, num_segments), math.min(f2, num_segments)
            segments[j] = torch.Tensor({f1, f2})
        end
        --print(segments)

        local x, acc_x = dh:plot_idt_density_seg(video, segments)
        local acc_p = acc_x:double():div(acc_x[acc_x:size(1)])
        f:write(video_name, acc_p)
    end
    while true do
        print("Iter: ", iter)
        iter = iter + 1
        local video_name, infos, video, labels, segments = dh:getTestSample()

        if video_name == nil then
            break
        end

        local num_segments = video:size(1)
        local num_frames = infos[3]
        for j = 1, segments:size(1) do
            local f1, f2 = torch.round(segments[j][1] * num_segments), torch.round(segments[j][2] * num_segments)
            f1, f2 = math.max(1, f1), math.max(1, f2)
            f1, f2 = math.min(f1, num_segments), math.min(f2, num_segments)
            segments[j] = torch.Tensor({f1, f2})
        end
        --print(segments)

        local x, acc_x = dh:plot_idt_density_seg(video, segments)
        local acc_p = acc_x:double():div(acc_x[acc_x:size(1)])
        f:write(video_name, acc_p)
    end
    f:close()
end

function static_action_bound()
    local dh = ThumosDataSet()
    local a, b, c, d, e, f = 0, 0, 0, 0, 0, 0
    while true do

        local video_name, infos, video, labels, segments = dh:getTestSample()
        if video_name == nil then
            break
        end
        for  j = 1, segments:size(1) do
            local st, ed = segments[j][1], segments[j][2]
            local diff = ed - st
            if diff >= 0.0 and diff < 0.01 then
                a = a + 1
            elseif diff >= 0.01 and diff < 0.02 then
                b = b + 1
            elseif diff >= 0.02 and diff < 0.03 then
                c = c + 1
            elseif diff >= 0.03 and diff < 0.04 then
                d = d + 1
            elseif diff >= 0.04 and diff < 0.05 then
                e = e + 1
            else
                f = f + 1
            end
        end
    end
    print('0.00 - 0.01: ', a)
    print('0.01 - 0.02: ', b)
    print('0.02 - 0.03: ', c)
    print('0.03 - 0.04: ', d)
    print('0.04 - 0.05: ', e)
    print('> 0.05: ', f)
    print('total: ', a + b + c + d + e + f)
end

function static_action_bound_cls()
    local dh = ThumosDataSet()
    local cnt = {}
    local ins_cnt = torch.zeros(20)
    while true do

        local video_name, infos, video, labels, segments = dh:getTestSample()
        if video_name == nil then
            break
        end
        for  j = 1, segments:size(1) do
            local st, ed = segments[j][1], segments[j][2]
            local diff = ed - st
            local l = labels[j]
            if cnt[l] == nil then cnt[l] = {} end
            table.insert(cnt[l], diff)
            ins_cnt[l] = ins_cnt[l] + 1
        end
    end
    print(ins_cnt)
    print(ins_cnt:sum())
    for i = 1, 20 do
        --print(cnt[i])
        --[[
        print(i, dh.label2name[i])
        local total = 0
        for j = 1, #cnt[i] do
            total = total + cnt[i][j]
        end
        print('Average: ', total/#cnt[i])
        ]]

        --io.read()
    end
end
--static_idts_density()
--static_action_bound()
static_action_bound_cls()

