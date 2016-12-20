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
cmd:option('--predPath', '', 'path to a previously saved model')
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
    --classifier = torch.load("/home/amax/cxt/thumos14/softmax3/trained_classifier_300000.t7")  -- trained on my ubuntu, no background
    classifier = torch.load("/home/amax/cxt/thumos14/softmax4/trained_classifier_400000.t7")
end

function _clip_segments(segments)
    segments[{{}, {1}}]:cmax(0)
    segments[{{}, {1}}]:cmin(1)
    segments[{{}, {2}}]:cmax(0)
    segments[{{}, {2}}]:cmin(1)
    segments[{{}, {1}}]:cmin(segments[{{}, {2}}])
    return segments
end

function apply_nms(seg_pred, cls_pred, scores, num_classes)
    local seg_pred_set = {}
    local scores_set = {}
    for i = 1, #cls_pred do
        if seg_pred_set[cls_pred[i]] == nil then
            seg_pred_set[cls_pred[i]] = {}
            scores_set[cls_pred[i]] = {}
        end
        table.insert(seg_pred_set[cls_pred[i]], seg_pred[i])
        table.insert(scores_set[cls_pred[i]], scores[i])
    end
    local nms_seg = {}
    local nms_label = {}
    local nms_scores = {}
    for i = 1, num_classes do
        if seg_pred_set[i] ~= nil and #seg_pred_set[i] > opt.abd then
            local seg, s = torch.Tensor(seg_pred_set[i]), torch.Tensor(scores_set[i])
            local pick = utils.nms(seg, s, opt.nms)
            seg = seg:index(1, pick)
            s = s:index(1, pick)
            for j = 1, pick:size(1) do
                table.insert(nms_seg, seg[j])
                table.insert(nms_scores, s[j])
                table.insert(nms_label, i)
            end
        end
    end
    return nms_seg, nms_label, nms_scores
end

-- mode 0
function demo()
    --agent:evaluate()
    classifier:evaluate()

    local softmax = nn.SoftMax()
    local dh = ThumosDataSet()
    while true do

        local video_name, infos, video, labels, segments = dh:getTestSample()

        if video_name == nil then
            break
        end
        print(labels)
        print(segments)
        local rnn_output = agent:forward({video})[1]
        local pred = torch.Tensor(#rnn_output, 2)
        for j = 1, #rnn_output do
            print("step: ", j)
            pred[j] = _clip_segments(rnn_output[j])
            local idt = dh:get_idt_seg_mid(video, pred[j])
            idt = idt:cuda()
            local cls_pred = classifier:forward(idt)
            cls_pred = cls_pred:double()
            cls_pred = softmax:forward(cls_pred)
            local maxVal, maxId = torch.max(cls_pred, 2)
            if maxVal[1][1] >= opt.conf then
                print(">> Yes <<")
                print(maxId[1][1], maxVal[1][1])
                print(pred[j])
            else
                print("No Detection")
                print(maxId[1][1], maxVal[1][1])
                print(pred[j])
            end
        end
        print("\n")
        io.read()
    end
end
--demo()

-- mode 1
function test_net()
    local f = io.open(opt.predPath, 'w')

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

        local seg_pred_set = {}
        local cls_pred_set = {}
        local scores = {}

        local rnn_output = agent:forward({video})[1]
        local pred = torch.Tensor(#rnn_output, 2)
        for j = 1, #rnn_output do
            pred[j] = _clip_segments(rnn_output[j])
            local idt = dh:get_idt_seg_mid(video, pred[j])
            idt = idt:cuda()
            local cls_pred = classifier:forward(idt)
            cls_pred = cls_pred:double()
            cls_pred = softmax:forward(cls_pred)
            local maxVal, maxId = torch.max(cls_pred, 2)
            if maxVal[1][1] > opt.conf then
                table.insert(seg_pred_set, {pred[j][1], pred[j][2]})
                table.insert(cls_pred_set, maxId[1][1])
                table.insert(scores, maxVal[1][1])
            end
        end

        seg_pred_set, cls_pred_set, scores = apply_nms(seg_pred_set, cls_pred_set, scores, opt.num_classes)  -- apply nms 
        for j = 1, #seg_pred_set do
            line = video_name .. '\t' .. string.format('%.1f', seg_pred_set[j][1] * duration) .. '\t' .. string.format('%.1f', seg_pred_set[j][2] * duration)
                .. '\t' .. cls_pred_set[j] .. '\t' .. tostring(scores[j]) .. '\n'
            --print(line)
            f:write(line)
        end
        --print('========================================\n\n')
        --io.read()

    end
    f:close()
    print("Done...")
end

-- mode 2
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
function test_net_idtdensity()
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
        local num_frames = infos[3]

        print(labels)
        print(segments*duration)

        local seg_pred_set = {}
        local cls_pred_set = {}
        local scores = {}

        local density, _ = dh:plot_idt_density(video)
        for i = 5, density:size(1)-4 do
            local num_idt = 0
            for j = i-4, i+4, 2 do
                num_idt = num_idt + video[j][{{1, 4000}}]:sum()
            end
            if density[i] > density[i-1] and density[i] > density[i+1] and num_idt/5.0 > 1000 then
                local l = (16 * i) / num_frames
                local pred = torch.Tensor({math.max(0.0, l-0.0065), math.min(duration, l+0.0065)})
                
                local idt = torch.zeros(16000)
                for j = i-4, i+4, 2 do
                    idt = idt + video[j]
                end
                idt:div(num_idt)
                idt = idt:cuda()
                local cls_pred = classifier:forward(idt)
                cls_pred = cls_pred:double()
                cls_pred = softmax:forward(cls_pred)
                

                --[[
                local cls_pred = torch.zeros(20)
                local non_zero = 0
                for j = i-1, i+1 do 
                    if density[j] > 0 then
                        local idt = video[j]:div(density[j])
                        idt = idt:cuda()
                        local cls_sin = classifier:forward(idt)
                        cls_sin = cls_sin:double()
                        cls_sin = softmax:forward(cls_sin)
                        cls_pred = cls_pred + cls_sin
                        non_zero = non_zero + 1
                    end
                end
                cls_pred:div(non_zero)
                ]]
                local maxVal, maxId = torch.max(cls_pred, 1)
                print(maxVal[1], maxId[1])
                io.read()
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
            print(line)
            f:write(line)
        end
        --io.read()

    end


    f:close()
    print("Done...")
end

-- mode 4 : score every frame
function test_net_sf()
    local f = io.open(opt.predPath, 'w')

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
        local num_frames = infos[3]

        --print(labels)
        --print(segments*duration)
        local seg_pred_set = {}
        local cls_pred_set = {}
        local scores = {}

        for i = 1, segments:size(1) do
            idt = dh:get_idt_seg_mid(video, segments[i])
            local num_idt = idt[{{1, 4000}}]:sum()
            idt:div(num_idt)
            idt = idt:cuda()
            local cls_pred = classifier:forward(idt)
            cls_pred = cls_pred:double()
            cls_pred = softmax:forward(cls_pred)
            

            local maxVal, maxId = torch.max(cls_pred, 1)
            if maxVal[1] > opt.conf then
                table.insert(seg_pred_set, {segments[i][1], segments[i][2]})
                table.insert(cls_pred_set, maxId[1])
                table.insert(scores, maxVal[1])
            end
        end
        for j = 1, #seg_pred_set do
            line = video_name .. '\t' .. string.format('%.1f', seg_pred_set[j][1] * duration) .. '\t' .. string.format('%.1f', seg_pred_set[j][2] * duration)
                .. '\t' .. cls_pred_set[j] .. '\t' .. tostring(scores[j]) .. '\n'
            --print(line)
            f:write(line)
        end
        --[[


        local density, _ = dh:plot_idt_density(video)
        local l_set = {0.007, 0.02}
        for _, s in pairs(l_set) do
            local seg_pred_set = {}
            local cls_pred_set = {}
            local scores = {}
            for i = 2, density:size(1)-1 do
                if density[i] > density[i-1] and density[i] > density[i+1] and density[i] > 1000 then
                    local l = (16 * i) / num_frames
                    local pred = torch.Tensor({math.max(0.0, l-s), math.min(duration, l+s)})
                    
                    local idt = dh:get_idt_seg(video, pred)
                    local num_idt = idt[{{1, 4000}}]:sum()
                    idt:div(num_idt)
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

            end

            seg_pred_set, cls_pred_set, scores = apply_nms(seg_pred_set, cls_pred_set, scores, opt.num_classes)  -- apply nms 
            for j = 1, #seg_pred_set do
                line = video_name .. '\t' .. string.format('%.1f', seg_pred_set[j][1] * duration) .. '\t' .. string.format('%.1f', seg_pred_set[j][2] * duration)
                    .. '\t' .. cls_pred_set[j] .. '\t' .. tostring(scores[j]) .. '\n'
                --print(line)
                f:write(line)
            end
        end
        ]]
        --io.read()

    end


    f:close()
    print("Done...")
end

function test_svmnet_global_nms()
    local pf = io.open(opt.predPath, 'w')
    assert(pf)
    local jsonTable = {}
    local results = {}
    local external_data = {}
    jsonTable["version"] = "VERSION 1.3"

    agent:evaluate()
    local softmax = nn.SoftMax()
    local svms = {}
    local model_name = {}
    local f = io.open('./all/svms/svm_models.txt', 'r')
    assert(f)
    local cnt = 1
    for line in f:lines() do
        local re = utils.split(line, ',')
        model_name[cnt] = re[1]
        print(model_name[cnt] .. ' loaded!')
        svms[cnt] = torch.load(model_name[cnt])
        svms[cnt]:evaluate()
        cnt = cnt + 1
    end
    f:close()

    local dh = SingleDataSet()
    local _, valSet, _, _ = dh:setClass("Root", "validation")
    local label2name, name2label, all2sub, sub2all = dh:getMapping()
    local randIdx = torch.randperm(#valSet)
    for i = 1, #valSet do
        local vidName = valSet[randIdx[i]]
        local info = dh:getVideoInfo(vidName)
        local duration = info["duration"]
        print(i, vidName)
        local video, label_target, segment_target = dh:getSingleSample(valSet[randIdx[i]])
        for j = 1, label_target:size(1) do
            label_target[j] = all2sub[label_target[j]]
            --print(label_target[j], label2name[label_target[j]])
        end
        --print(segment_target)
        local rnn_output = rnn:forward({video})
        local seg_pred_set = {}
        local cls_pred_set = {}
        local scores = {}
        local pred_set = {}
        for j = 1, #rnn_output do
            local output = agent:forward(rnn_output[j])
            seg_pred = _clip_segments(output[1]):resize(2)
            local c3d, _, _ = dh:getSingleTrimedByName(valSet[randIdx[i]], seg_pred, 10)

            -- Use SVM Classifier
            local cls_pred = torch.zeros(201)

            for c = 1, 201 do
                local svm_output = svms[c]:forward(c3d)
                cls_pred[c] = svm_output:sum() / 10
            end
            --[[
            for k = 1, c3d:size(1) do
                for c = 1, 201 do
                    local svm_output = svms[c]:forward(c3d[k])
                    cls_pred[c] = cls_pred[c] + svm_output[1]
                end
            end
            ]]

            for k = 1, cls_pred:size(1) do
                if cls_pred[k] ~= cls_pred[k] then
                    cls_pred[k] = -1000.0
                end
            end

            cls_pred = softmax:forward(cls_pred)
            local maxVal, maxId = torch.max(cls_pred, 1)

            --print(maxId[1], maxVal[1])
            if maxVal[1] >= 0.6 and maxId[1] ~= 201 then
            --if maxId[1] ~= 201 then
                table.insert(seg_pred_set, {seg_pred[1], seg_pred[2]})
                table.insert(cls_pred_set, maxId[1])
                table.insert(scores, maxVal[1])
            end
        end
        seg_pred_set, cls_pred_set, scores = apply_nms(seg_pred_set, cls_pred_set, scores, 201)  -- apply nms 
        for j = 1, #seg_pred_set do
            local pred = {}
            pred["score"] = scores[j]
            pred["segment"] = {seg_pred_set[j][1] * duration, seg_pred_set[j][2] * duration}
            pred["label"] = label2name[cls_pred_set[j]]
            --print(label2name[cls_pred_set[j]])
            --print(cls_pred_set[j], pred["score"])
            --print(seg_pred_set[j])
            table.insert(pred_set, pred)
        end
        --print('========================================\n')
        --io.read()

        results[dh:cut_name(vidName)] = pred_set
    end
    jsonTable["results"] = results
    
    external_data["used"] = false
    external_data["details"] = "This is a fake submission for the validation subset."
    jsonTable["external_data"] = external_data

    local json_text = cjson.encode(jsonTable)
    pf:write(json_text)
    pf:close()
    print("Done...")
end

if opt.mode == 0 then
    demo()
elseif opt.mode == 1 then
    test_net()
elseif opt.mode == 2 then
    test_net_slidingwindow()
elseif opt.mode == 3 then
    test_net_idtdensity()
elseif opt.mode == 4 then
    test_net_sf()
else
    test_svmnet_global_nms()
end

