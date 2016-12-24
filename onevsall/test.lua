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
cmd:option('--conf', 0.5, 'confidence for prediction')
cmd:option('--num_classes', 26, 'number of classes')
cmd:option('--label', 1, 'class to be test')

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
    agent = torch.load("./model/agent_1_200.t7")
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


function test_single_net()
    local f = io.open(opt.predPath, 'w')
    local jsonTable = {}
    local results = {}
    local external_data = {}
    jsonTable["version"] = "VERSION 1.3"

    --agent:evaluate()
    local dh = SingleDataSet()
    dh:setClass("Playing sports", "validation")
    local valSet, _ = dh:separateOneVsAll(opt.label)
    local label2name, name2label, all2sub, sub2all = dh:getMapping()
    local randIdx = torch.randperm(#valSet)
    print(label2name[opt.label])
    --io.read()

    local average_prec, average_recall = 0.0, 0.0
    for i = 1, #valSet do
        local vid = valSet[randIdx[i]]
        local vidName = vid['name']
        --print(i, vidName)
        local duration = vid["duration"]
        --print(info)
        local video, label_target, segment_target = dh:getSingleSample(vidName)
        --print(all2sub[label_target[1]])
        --print(segment_target)
        local rnn_output = agent:forward({video})[1]
        local seg_pred_set = {}
        local cls_pred_set = {}
        local scores = {}
        local pred_set = {}
        for j = 1, #rnn_output do
            local cls_pred, seg_pred = unpack(rnn_output[j])
            seg_pred = _clip_segments(seg_pred):resize(2)
            --[[
            print("Prediction: ", j)
            print(cls_pred[1][1])
            print(seg_pred)
            io.read()
            ]]
            if cls_pred[1][1] > 0 and seg_pred[2] > seg_pred[1] then
                table.insert(seg_pred_set, {seg_pred[1], seg_pred[2]})
                --table.insert(cls_pred_set, opt.label)
                --table.insert(scores, 1.0)
            end
        end
        if #seg_pred_set == 0 then
            rand = torch.rand(1)[1]
            table.insert(seg_pred_set, {rand, rand+0.5})
        end
        proposals = torch.Tensor(seg_pred_set)
        --print(proposals)
        ov = utils.interval_overlap(segment_target, proposals)
        maxOv, maxId = torch.max(ov, 1)
        tp = maxOv:ge(0.5):sum()
        ids = maxId[maxOv:ge(0.5)]
        tf = 0
        if ids:dim() > 0 then
            set = {}
            for i = 1, ids:size(1) do
                if set[ids[i]] == nil then
                    set[ids[i]] = true
                    tf = tf+1
                end
            end
        end
        fn = segment_target:size(1) - tf
        total = proposals:size(1)
        prec = tp / total
        recall = tf / (tf + fn)
        average_prec = average_prec + prec
        average_recall = average_recall + recall
        --print(tp, total, tf, tf+fn)
        --print(string.format('precision: %.3f, recall: %.3f', prec, recall))
       
        --[=[
        for j = 1, #seg_pred_set do
            local pred = {}
            pred["score"] = scores[j]
            pred["segment"] = {seg_pred_set[j][1] * duration, seg_pred_set[j][2] * duration}
            pred["label"] = label2name[cls_pred_set[j]]
            --[[
            print(cls_pred_set[j], pred["score"])
            print(seg_pred_set[j])
            ]]
            table.insert(pred_set, pred)
        end
        ]=]
        --print('========================================\n\n')
        --io.read()


        results[dh:cut_name(vidName)] = pred_set
    end
    print(string.format('precision: %.3f, recall: %.3f', average_prec/#valSet, average_recall/#valSet))
    

    jsonTable["results"] = results
    
    external_data["used"] = false
    external_data["details"] = "This is a fake submission for the validation subset."
    jsonTable["external_data"] = external_data

    local json_text = cjson.encode(jsonTable)
    f:write(json_text)
    f:close()
    print("Done...")
end

function test_net()
    -- Load svm models
    local svms_path = {
        'model/agent_1_200.t7',
        'model/agent_2_200.t7',
        'model/agent_3_200.t7',
        'model/agent_4_440.t7',
        'model/agent_5_500.t7',
        'model/agent_6_200.t7',
        'model/agent_7_200.t7',
        'model/agent_8_200.t7',
        'model/agent_9_500.t7',
        'model/agent_10_800.t7',
        'model/agent_11_500.t7',
        'model/agent_12_700.t7',
        'model/agent_13_600.t7',
        'model/agent_14_200.t7',
        'model/agent_15_500.t7',
        'model/agent_16_200.t7',
        'model/agent_17_600.t7',
        'model/agent_18_240.t7',
        'model/agent_19_400.t7',
        'model/agent_20_300.t7',
        'model/agent_21_600.t7',
        'model/agent_22_200.t7',
        'model/agent_23_700.t7',
        'model/agent_24_440.t7',
        'model/agent_25_400.t7',
        'model/agent_26_400.t7',
    }
    local svms = {}
    for i = 1, 26 do
        svms[i] = torch.load(svms_path[i])
        print('OneVsAll model ' .. svms_path[i] .. ' loaded')
    end

    local f = io.open(opt.predPath, 'w')
    local jsonTable = {}
    local results = {}
    local external_data = {}
    jsonTable["version"] = "VERSION 1.3"

    --agent:evaluate()
    local dh = SingleDataSet()
    local _, valSet, _, _ = dh:setClass("Playing sports", "validation")
    local label2name, name2label, all2sub, sub2all = dh:getMapping()
    local randIdx = torch.randperm(#valSet)

    for i = 1, #valSet do
        local vidName = valSet[randIdx[i]]
        local info = dh:getVideoInfo(vidName)
        print(i, vidName)
        local duration = info["duration"]
        local video, label_target, segment_target = dh:getSingleSample(vidName)
        for j = 1, label_target:size(1) do
            label_target[j] = all2sub[label_target[j]]
        end
        --print(label_target)
        --print(segment_target)
        --io.read()
        
        local cls_pred_set = {}
        local seg_pred_set = {}
        local scores = {}
        local pred_set = {}
        for j = 1, 26 do
            --print('label: ', j)
            local rnn_output = svms[j]:forward({video})[1]
            for k = 1, #rnn_output do
                local cls_pred, seg_pred = unpack(rnn_output[k])
                seg_pred = _clip_segments(seg_pred):resize(2)
                if cls_pred[1][1] > 1.0 and seg_pred[2] > seg_pred[1] then
                    --print(seg_pred[1], seg_pred[2], cls_pred[1][1])
                    table.insert(seg_pred_set, {seg_pred[1], seg_pred[2]})
                    table.insert(cls_pred_set, j)
                    table.insert(scores, cls_pred[1][1])
                end
            end
            --io.read()
        end

        ---- Get top N -----------------------
        --[=[
        if #scores  > 0  then
            local seg_pred_saved, cls_pred_saved, scores_saved = {}, {}, {}
            local sorted_scores, inds = torch.sort(torch.Tensor(scores), true)
            for j = 1, inds:size(1) do
                if j > 3 then break end
                table.insert(seg_pred_saved, seg_pred_set[inds[j]])
                table.insert(cls_pred_saved, cls_pred_set[inds[j]])
                table.insert(scores_saved, scores[inds[j]])
            end
            cls_pred_set, seg_pred_set, scores = cls_pred_saved, seg_pred_saved, scores_saved
        end
        ]=]

       
        seg_pred_set, cls_pred_set, scores = apply_nms(seg_pred_set, cls_pred_set, scores, opt.num_classes)
        for j = 1, #seg_pred_set do
            local pred = {}
            pred["score"] = scores[j]
            pred["segment"] = {seg_pred_set[j][1] * duration, seg_pred_set[j][2] * duration}
            pred["label"] = label2name[cls_pred_set[j]]
            --print(cls_pred_set[j], pred["score"])
            --print(seg_pred_set[j])
            table.insert(pred_set, pred)
        end
        --print('========================================\n\n')
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

if opt.mode == 0 then
    test_single_net()
else
    test_net()
end



