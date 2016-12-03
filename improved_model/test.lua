--require 'torch'
--require 'nn'
require 'dp'
require 'rnn'
require 'VolumetricGlimpse'
require 'DetReward'
require 'DetLossCriterion'
require 'SingleDataSet'
require 'MyRecurrentAttention'
require 'nngraph'
require 'MyConstant'
require 'utils'

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
cmd:option('--num_classes', 200, 'number of classes')

cmd:text()
local opt = cmd:parse(arg or {})
if not opt.silent then
   table.print(opt)
end

--[[Model]]--
if opt.rnnPath ~= '' and opt.agentPath ~= '' and clsPath ~= '' then
    assert(paths.filep(opt.rnnPath), opt.rnnPath ..' does not exist')
    assert(paths.filep(opt.agentPath), opt.agentPath ..' does not exist')
    rnn = torch.load(opt.rnnPath)
    agent = torch.load(opt.agentPath)
else
    agent = torch.load("./detector_1_1/trained_agent_40000.t7")
end

if opt.clsPath ~= '' then
    assert(paths.filep(opt.clsPath), opt.clsPath ..' does not exist')
    classifier = torch.load(opt.clsPath)
else
    classifier = torch.load("../separate_train/all/classifier3/trained_classifier_120000.t7")  -- trained on my ubuntu, no background
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
        if seg_pred_set[i] ~= nil then
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

function demo()
    --agent:evaluate()
    classifier:evaluate()

    local softmax = nn.SoftMax()
    local dh = SingleDataSet()
    local _, valSet, _, _ = dh:setClass("Root", "validation")
    local label2name, name2label, all2sub, sub2all = dh:getMapping()
    local randIdx = torch.randperm(#valSet)
    for i = 1, #valSet do
        --print(valSet[randIdx[i]])
        local vidName = valSet[randIdx[i]]
        local info = dh:getVideoInfo(vidName)
        --print(info)
        local video, label_target, segment_target = dh:getSingleSample(vidName)
        for j = 1, label_target:size(1) do
            label_target[j] = all2sub[label_target[j]]
        end
        local output  = agent:forward({video})
        local pred = _clip_segments(output[1])
        local c3d, _, _ = dh:getSingleTrimedByName(vidName, pred[1])
        local cls_pred = classifier:forward(c3d)
        cls_pred = softmax:forward(cls_pred)
        local maxVal, maxId = torch.max(cls_pred, 1)

        print("Ground Truth: ")
        print(label_target)
        print(segment_target)
        print("Prediction: ")
        print(maxId[1], maxVal[1])
        print(pred[1])
        print("================================\n\n")
        io.read()
    end
end
--demo()

function test_net()
    local f = io.open(opt.predPath, 'w')
    local jsonTable = {}
    local results = {}
    local external_data = {}
    jsonTable["version"] = "VERSION 1.3"

    --agent:evaluate()
    classifier:evaluate()
    softmax = nn.SoftMax()
    local dh = SingleDataSet()
    local _, valSet, _, _ = dh:setClass("Root", "validation")
    local label2name, name2label, all2sub, sub2all = dh:getMapping()
    local randIdx = torch.randperm(#valSet)
    for i = 1, #valSet do
        local vidName = valSet[randIdx[i]]
        print(i, vidName)
        --vidName = "v_VdY1Shdks6o.mp4"
        local info = dh:getVideoInfo(vidName)
        local duration = info["duration"]
        local video, label_target, segment_target = dh:getSingleSample(vidName)
        for j = 1, label_target:size(1) do
            label_target[j] = all2sub[label_target[j]]
        end
        local output  = agent:forward({video})
        local seg_pred = _clip_segments(output[1])
        local c3d, _, _ = dh:getSingleTrimedByName(vidName, seg_pred[1])
        local cls_pred = classifier:forward(c3d)
        cls_pred = softmax:forward(cls_pred)
        local maxVal, maxId = torch.max(cls_pred, 1)
        local pred = {}
        pred["score"] = maxVal[1]
        pred["segment"] = {seg_pred[1][1] * duration, seg_pred[1][2] * duration}
        pred["label"] = label2name[maxId[1]]

        local pred_set = {pred}

        results[dh:cut_name(vidName)] = pred_set
    end
    jsonTable["results"] = results
    
    external_data["used"] = false
    external_data["details"] = "This is a fake submission for the validation subset."
    jsonTable["external_data"] = external_data

    local json_text = cjson.encode(jsonTable)
    f:write(json_text)
    f:close()
    print("Done...")
end


function test_net_nms()
    local f = io.open(opt.predPath, 'w')
    local jsonTable = {}
    local results = {}
    local external_data = {}
    jsonTable["version"] = "VERSION 1.3"

    rnn = agent:get(1)
    regression = agent:get(3)
    print(rnn)
    print(regression)
    regression:evaluate()
    classifier:evaluate()
    softmax = nn.SoftMax()

    local dh = SingleDataSet()
    local _, valSet, _, _ = dh:setClass("Root", "validation")
    local label2name, name2label, all2sub, sub2all = dh:getMapping()
    local randIdx = torch.randperm(#valSet)
    for i = 1, #valSet do
        local vidName = valSet[randIdx[i]]
        local info = dh:getVideoInfo(vidName)
        local duration = info["duration"]
        print(i, vidName)
        local video, label_target, segment_target = dh:getSingleSample(vidName)
        for j = 1, label_target:size(1) do
            label_target[j] = all2sub[label_target[j]]
        end
        --print(label_target)
        --print(segment_target)
        local rnn_output = rnn:forward({video})
        local seg_pred_set = {}
        local cls_pred_set = {}
        local scores = {}
        local pred_set = {}
        for j = 1, #rnn_output do
            local output = regression:forward(rnn_output[j])
            seg_pred = _clip_segments(output):resize(2)
            local c3d, _, _ = dh:getSingleTrimedByName(valSet[randIdx[i]], seg_pred)
            local cls_pred = classifier:forward(c3d)
            cls_pred = softmax:forward(cls_pred)
            local maxVal, maxId = torch.max(cls_pred, 1)
            if maxVal[1] >= 0.45 then
                table.insert(seg_pred_set, {seg_pred[1], seg_pred[2]})
                table.insert(cls_pred_set, maxId[1])
                table.insert(scores, maxVal[1])
            end
        end
        seg_pred_set, cls_pred_set, scores = apply_nms(seg_pred_set, cls_pred_set, scores, opt.num_classes)  -- apply nms 
        for j = 1, #seg_pred_set do
            local pred = {}
            pred["score"] = scores[j]
            pred["segment"] = {seg_pred_set[j][1] * duration, seg_pred_set[j][2] * duration}
            pred["label"] = label2name[cls_pred_set[j]]
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
    f:write(json_text)
    f:close()
    print("Done...")
end

function test_svmnet_nms()
    local f = io.open(opt.predPath, 'w')
    local jsonTable = {}
    local results = {}
    local external_data = {}
    jsonTable["version"] = "VERSION 1.3"

    agent:evaluate()
    svms = {}
    local softmax = nn.SoftMax()
    model_name = {
        "./sports/svms/1_svm_5000.t7",
        "./sports/svms/2_svm_1000.t7",
        "./sports/svms/3_svm_1000.t7",
        "./sports/svms/4_svm_1000.t7",
        --"./sports/svms/5_svm_1000.t7",
        "./sports/advanced_svms/5_svm_200.t7",
        "./sports/svms/6_svm_1000.t7",
        "./sports/svms/7_svm_3000.t7",
        "./sports/svms/8_svm_500.t7",
        "./sports/svms/9_svm_500.t7",
        "./sports/svms/10_svm_500.t7",
        "./sports/svms/11_svm_500.t7",
        --"./sports/svms/12_svm_500.t7",
        "./sports/advanced_svms/12_svm_800.t7",
        "./sports/svms/13_svm_500.t7",
        "./sports/svms/14_svm_1000.t7",
        "./sports/svms/15_svm_500.t7",
        "./sports/svms/16_svm_500.t7",
        "./sports/svms/17_svm_1000.t7",
        --"./sports/svms/18_svm_500.t7",
        "./sports/advanced_svms/18_svm_200.t7",
        "./sports/svms/19_svm_500.t7",
        "./sports/svms/20_svm_1000.t7",
        "./sports/svms/21_svm_1000.t7",
        "./sports/svms/22_svm_500.t7",
        "./sports/svms/23_svm_2000.t7",
        "./sports/svms/24_svm_1000.t7",
        "./sports/svms/25_svm_500.t7",
        "./sports/svms/26_svm_5000.t7",
        "./sports/svms/27_svm_80000.t7"
    }
    for i = 1, 27 do
        svms[i] = torch.load(model_name[i])
        svms[i]:evaluate()
    end

    local dh = SingleDataSet()
    local _, valSet, _, _ = dh:setClass("Playing sports", "validation")
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
        end
        print(label_target)
        print(segment_target)
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
            local cls_pred = torch.zeros(27)
            for k = 1, c3d:size(1) do
                for c = 1, 27 do
                    local svm_output = svms[c]:forward(c3d[k])
                    cls_pred[c] = cls_pred[c] + svm_output[1]
                end
            end

            cls_pred = softmax:forward(cls_pred)
            local maxVal, maxId = torch.max(cls_pred, 1)

            if maxVal[1] >= 0.6 and maxId[1] ~= 27 then
                table.insert(seg_pred_set, {seg_pred[1], seg_pred[2]})
                table.insert(cls_pred_set, maxId[1])
                table.insert(scores, maxVal[1])
            end
        end
        seg_pred_set, cls_pred_set, scores = apply_nms(seg_pred_set, cls_pred_set, scores, opt.num_classes)  -- apply nms 
        for j = 1, #seg_pred_set do
            local pred = {}
            pred["score"] = scores[j]
            pred["segment"] = {seg_pred_set[j][1] * duration, seg_pred_set[j][2] * duration}
            pred["label"] = label2name[cls_pred_set[j]]
            print(cls_pred_set[j], pred["score"])
            print(seg_pred_set[j])
            table.insert(pred_set, pred)
        end
        print('========================================\n')
        io.read()

        results[dh:cut_name(vidName)] = pred_set
    end
    jsonTable["results"] = results
    
    external_data["used"] = false
    external_data["details"] = "This is a fake submission for the validation subset."
    jsonTable["external_data"] = external_data

    local json_text = cjson.encode(jsonTable)
    f:write(json_text)
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
    test_net_nms()
elseif opt.mode == 3 then
    test_svmnet_nms()
else
    test_svmnet_global_nms()
end

