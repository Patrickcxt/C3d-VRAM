require 'nn'
require 'dp'
require 'SingleDataSet'
local cjson = require 'cjson'
--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Test a classifier Model')
cmd:text('Example:')
cmd:text('$> th test.lua > results.txt')
cmd:text('Options:')
cmd:option('--class', 1, 'svm for class to test')
cmd:option('--xpPath', '', 'path to a previously saved model')
cmd:option('--predPath', '', 'path to a previously saved model')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--mode', 0, 'demo, test_net or test_net2')
cmd:option('--silent', false, 'dont print anything to stdout')

cmd:text()
local opt = cmd:parse(arg or {})
if not opt.silent then
   table.print(opt)
end


--[[Model]]--
if opt.xpPath ~= '' then
    assert(paths.filep(opt.xpPath), opt.xpPath ..' does not exist')
    trained_model = torch.load(opt.xpPath)
else
    trained_model = torch.load('/media/chenxt/TOURO Mobile USB3.0/Action/C3d-VRAM/separate_train/sports/global_svms/1_svm_2000.t7')
end

function testSingleClass()
    trained_model:evaluate()
    local dh = SingleDataSet()
    dh:setClass("Playing sports", "validation")
    local valSet, _ = dh:seperateOneVsAllGlobal(opt.class, "validation")
    local randIdx = torch.randperm(#valSet)
    local label2name, name2label, all2sub, sub2all = dh:getMapping()
    print("Class Name: ", label2name[opt.class])

    local correct = 0
    local total = 0
    for i = 1, #valSet do
        print("Iter: ", total)
        local video, label_target, _ = dh:getSingleSample(valSet[randIdx[i]], 1)
        local output = trained_model:forward(video)
        --print("output: ", output[1])
        --io.read()
        if output[1] >= 0 then
            correct  = correct + 1
        end
            
        --print("==============================")
        total = total + 1
    end
    print("total: " , total)
    print("correct: " , correct)
    print("Correct Ratio: ", correct / total)
end


function testAllClass()
    svms = {}
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
        "./sports/svms/26_svm_500.t7",
        "./sports/svms/27_svm_80000.t7"
    }
    for i = 1, 27 do
        svms[i] = torch.load(model_name[i])
        svms[i]:evaluate()
    end

    local dh = SingleDataSet()
    dh:setClass("Playing sports", "validation")
    --dh:seperateOneVsAll(opt.class)
    local label2name, name2label, all2sub, sub2all = dh:getMapping()
    
    local correct = 0
    local total = 0
    
    while true do
        print("Iter: ", total)
        local video, label_target, _ = dh:getSingleTrimedSample()
        if video == nil then
            break
        end
        local scores = torch.zeros(27)
        for i = 1,  video:size(1) do
            for j = 1, 27 do
                local output = svms[j]:forward(video[i])
                scores[j] = scores[j] +  output[1]
            end
        end
        --print("score: ", scores)
        maxVal, maxId = torch.max(scores, 1)
        --[[
        print("GT Label:")
        print(label_target)
        print("Pred Label:")
        print(maxId[1])
        ]]
        if maxId[1] == label_target then
            correct  = correct + 1
        end
        total = total + 1
            
        --print("==============================")
    end
    print("total: " , total)
    print("correct: " , correct)
    print("Correct Ratio: ", correct / total)
end


if opt.mode == 0 then
    testSingleClass()
else
    testAllClass()
end




