--require 'torch'
--require 'nn'
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
--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Recurrent Model for Visual Attention')
cmd:text('Example:')
cmd:text('$> th test.lua > results.txt')
cmd:text('Options:')
cmd:option('--rnnPath', '', 'path to a previously saved model')
cmd:option('--agentPath', '', 'path to a previously saved model')
cmd:option('--clsPath', '', 'path to a previously saved model')
cmd:option('--predPath', 'tmp.txt', 'path to a previously saved model')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--mode', 0, 'demo, test_net or test_net_nms')
cmd:option('--numframes', 10, 'number of c3d segments for svm classifier')
cmd:option('--nms', 0.5, 'overlap of nms')
cmd:option('--conf', 0.5, 'confidence for prediction')
cmd:option('--num_classes', 20, 'number of classes')
cmd:option('--abd', 1, 'abondoned')

cmd:text()
local opt = cmd:parse(arg or {})
if not opt.silent then
   table.print(opt)
end

--[[Model]]--
if opt.agentPath ~= '' then
    assert(paths.filep(opt.agentPath), opt.agentPath ..' does not exist')
    agent = torch.load(opt.agentPath)
else
    agent = torch.load("./detector/trained_agent_5000.t7")
end

if opt.clsPath ~= '' then
    assert(paths.filep(opt.clsPath), opt.clsPath ..' does not exist')
    classifier = torch.load(opt.clsPath)
else
    classifier = torch.load("/home/amax/cxt/thumos14/softmax4/trained_classifier_600000.t7")
end

function _clip_segments(segments)
    segments[{{}, {1}}]:cmax(0)
    segments[{{}, {1}}]:cmin(1)
    segments[{{}, {2}}]:cmax(0)
    segments[{{}, {2}}]:cmin(1)
    segments[{{}, {1}}]:cmin(segments[{{}, {2}}])
    return segments
end


function apply_nms(all_boxes, thresh)
    local num_classes = #all_boxes
    local num_videos = #all_boxes[1]
    local nms_boxes = {}
    for i = 1, num_classes do
        nms_boxes[i] = {}
    end

    for cls_ind = 1, num_classes do
        for v_ind = 1, num_videos do
            nms_boxes[cls_ind][v_ind] = torch.Tensor(0)
            local dets = all_boxes[cls_ind][v_ind]
            if dets:dim() > 0 and dets[1][1] ~= -1 then
                local keep = utils.nms(dets[{{}, {1, 2}}], dets[{{}, {3}}],  thresh)
                if keep:size(1) > 0 then
                    nms_boxes[cls_ind][v_ind] = dets:index(1, keep)
                end
            end
        end
    end
end


function test_net_slidingwindow()
    local f = io.open(opt.predPath, 'w')
    --local f = io.open(opt.predPath, 'a')

    --agent:evaluate()
    classifier:evaluate()

    local softmax = nn.SoftMax()
    local dh = ThumosDataSet()
    local iter = 0
    while true do
        local video_name, infos, video, labels, segments = dh:getTestSample()
        print("Iter: ", iter, video_name)
        iter = iter + 1

        if video_name == nil then
            break
        end
        local duration = infos[1]
        --print(labels)
        --print(segments)

        local seg_pred_set = {}
        local cls_pred_set = {}
        local scores = {}

        for l = 0.015, 0.015, 0.04 do
            local step = l / 2
            for st = 0.0, 1-l, step do
                local pred = torch.Tensor({st, st+l})
                --print(pred)
                local idt = dh:get_idt_seg_mid(video, pred)
                local num_idt = idt[{{1, 4000}}]:sum()
                if num_idt > 0 then
                    idt = idt:div(num_idt)
                end
                idt = idt:cuda()
                local cls_pred = classifier:forward(idt)
                cls_pred = cls_pred:double()
                cls_pred = softmax:forward(cls_pred)
                local maxVal, maxId = torch.max(cls_pred, 1)
                if maxVal[1] > opt.conf then
                    table.insert(seg_pred_set, {pred[1], pred[2]})
                    table.insert(cls_pred_set, maxId[1])
                    table.insert(scores, maxVal[1])
                end
            end

            --print('========================================\n\n')
            --io.read()
        end
        seg_pred_set, cls_pred_set, scores = apply_nms(seg_pred_set, cls_pred_set, scores, opt.num_classes)  -- apply nms 
        for j = 1, #seg_pred_set do
            line = video_name .. '\t' .. string.format('%.1f', seg_pred_set[j][1] * duration) .. '\t' .. string.format('%.1f', seg_pred_set[j][2] * duration)
                .. '\t' .. cls_pred_set[j] .. '\t' .. tostring(scores[j]) .. '\n'
            --print(line)
            f:write(line)
        end
        --io.read()

    end
    f:close()
    print("Done...")
end

-- mode 3

max_per_set = {41, 488, 106, 98, 217,
               138, 170, 388, 48, 36, 
               242, 135, 169, 142, 399, 
               144, 48, 141, 88, 120
}

function test_net_idtdensity()
    local f = io.open(opt.predPath, 'w')
    --local f = io.open(opt.predPath, 'a')

    --agent:evaluate()
    classifier:evaluate()

    local softmax = nn.SoftMax()
    local dh = ThumosDataSet()
    local iter = 0

    local videos = {}
    local num_videos = 212

    local max_per_video = 80
    --local max_per_set = 5 * 212
    local thresh = torch.ones(opt.num_classes) * -math.huge
    local top_scores = {}
    local all_boxes = {}
    for i = 1, opt.num_classes do
        top_scores[i] = torch.Tensor(1):fill(-math.huge)
        all_boxes[i] = {}
    end

    local v = 1
    while true do
        local video_name, infos, video, labels, segments = dh:getTestSample()
        table.insert(videos, video_name)
        print("Iter: ", iter, video_name)
        iter = iter + 1

        if video_name == nil then
            break
        end
        local duration = infos[1]
        local num_frames = infos[3]

        --print(labels:reshape(labels:size(1)))
        print(thresh)
        --print(segments*duration)

        local pred_set = {}
        for i = 1, opt.num_classes do
            pred_set[i] = torch.Tensor(1, 3):fill(-1)
        end

        local density, _ = dh:plot_idt_density(video)
        for i = 5, density:size(1)-4 do
            local num_idt = 0
            for j = i-4, i+4, 2 do
                num_idt = num_idt + video[j][{{1, 4000}}]:sum()
            end
            if density[i] > density[i-1] and density[i] > density[i+1] and num_idt/5.0 > 600 then
                local l = (16 * i) / num_frames
                local seg_pred = torch.Tensor({math.max(0.0, l-0.0075), math.min(duration, l+0.0075)})
                
                local idt = torch.zeros(16000)
                for j = i-4, i+4, 2 do
                    idt = idt + video[j]
                end
                idt:div(num_idt)
                idt = idt:cuda()
                local cls_pred = classifier:forward(idt)
                cls_pred = cls_pred:double()
                cls_pred = softmax:forward(cls_pred)
                
                local maxVal, maxId = torch.max(cls_pred, 1)
                local cls, score = maxId[1], maxVal[1]
                --print(cls, score)
                --io.read()

                if score > thresh[cls] then
                    local pred = torch.Tensor({{seg_pred[1], seg_pred[2], score}})
                    pred_set[cls] = pred_set[cls]:cat(pred, 1)
                end
            end
        end

        for i = 1, opt.num_classes do
            repeat
                if pred_set[i]:size(1) > 1 then 
                    pred_set[i] = pred_set[i][{{2, -1}, {}}]
                else
                    all_boxes[i][v] = pred_set[i]
                    break
                end
                local scores, inds = torch.sort(pred_set[i][{{}, {3}}], 1, true)
                local max_this_video = math.min(max_per_video, inds:size(1))
                scores = scores[{{1, max_this_video}, {}}]
                pred_set[i] = pred_set[i]:index(1, inds[{{1, max_this_video}, {}}]:reshape(max_this_video))

                -- push new scores to minheap
                top_scores[i] = top_scores[i]:cat(scores:reshape(max_this_video))
                if top_scores[i]:size(1) > max_per_set[i]*2 then
                    top_scores[i], _ = top_scores[i]:sort()
                    local st = top_scores[i]:size(1) - max_per_set[i]*2 + 1
                    top_scores[i] = top_scores[i][{{st, -1}}]
                    thresh[i] = top_scores[i][1]
                end

                all_boxes[i][v] = pred_set[i]
            until true
            --print(i, dh.label2name[i])
            --print(all_boxes[i][v])
            --io.read()
            
        end

        v = v + 1
        --io.read()

    end


    for j = 1, opt.num_classes - 1 do
        for i = 1, num_videos do
            local inds = all_boxes[j][i][{{}, {1}}]:gt(thresh[j])
            local num_keep = inds:sum()
            if num_keep == 0 then
                all_boxes[j][i] = torch.zeros(1, 3)
            else
                inds = inds:cat(inds):cat(inds)
                all_boxes[j][i] = all_boxes[j][i][inds]:reshape(num_keep, 3)
            end
        end
    end

    local nms_dets = apply_nms(all_boxes, opt.nms)
    --seg_pred_set, cls_pred_set, scores = apply_nms(seg_pred_set, cls_pred_set, scores, opt.num_classes)  -- apply nms
    for j = 1, num_videos do
        for i = 1, opt.num_classes do
            for k = 1, nms_dets:size(1) do
                line = videos[j] .. '\t' .. string.format('%.1f', nms_dets[i][j][k][1] * duration) .. '\t' .. string.format('%.1f', nms_dets[i][j][k][2] * duration)
                .. '\t' .. i .. '\t' .. tostring(det_nms[i][j][k][3] ) .. '\n'
                print(line)
                f:write(line)
            end
        end
    end

    f:close()
    print("Done...")
end


--test_net_slidingwindow()
test_net_idtdensity()

