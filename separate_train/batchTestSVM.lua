require 'nn'
require 'dp'
require 'SingleDataSet'
require 'utils'
local cjson = require 'cjson'
--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Test a classifier Model')
cmd:text('Example:')
cmd:text('$> th test.lua > results.txt')
cmd:text('Options:')
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

function testSingleClass(clsLabel)
    
    local dh = SingleDataSet()
    local maxRatio, maxIter = 0.0, 0
    local maxModelName = "./all/svms/" .. tostring(clsLabel) .. '_svm_200.t7'
    for iter = 52000, 100000, 2000 do
        local model_name = './all/svms/' .. tostring(clsLabel) .. '_svm_' .. tostring(iter) .. '.t7'
        if not utils.file_exists(model_name) then
            break
        end
        local trained_model = torch.load(model_name)
        trained_model:evaluate()

        dh:setClass("Root", "validation")
        dh:seperateOneVsAll(clsLabel)
        local label2name, name2label, all2sub, sub2all = dh:getMapping()
        print("iter: ", iter)
        print("Class Name: ", label2name[clsLabel])

        
        local correct = 0
        local total = 0
        while true do
            local video, label_target, _ = dh:getSingleTrimedSample(10)
            if video == nil then
                break
            end
            local scores = 0.0
            for i = 1,  video:size(1) do
                local output = trained_model:forward(video[i])
                scores = scores + output[1]
            end
            if scores >= 0 then
                correct  = correct + 1
            end
            total = total + 1
        end
        print("total: " , total)
        print("correct: " , correct)
        local ratio = correct / total
        print("Correct Ratio: ", ratio)
        if ratio > maxRatio then
            maxRatio = ratio
            maxIter = iter
            maxModelName = model_name
        end
        print('=============================================\n')
    end
    print('\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print("Max Iter: ", maxIter)
    print("Max Ratio: ", maxRatio)
    print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n')
    dh:destroy()
    f = io.open('./all/svms/advanced_models.txt', 'a+')
    local line = maxModelName .. ', ' .. tostring(maxRatio) .. '\n'
    f:write(line)
    f.close()

end

function testAllClass()
    local svms = {}
    local model_name = {}
    local f = io.open('./all/svms/svm_models.txt', 'r')
    local cnt = 1
    for line in f:lines() do
        local re = utils.split(line, ',')
        model_name[cnt] = re[1]
        print(model_name[cnt] .. ' loaded!')
        svms[cnt] = torch.load(model_name[cnt])
        svms[cnt]:evaluate()
        cnt = cnt + 1
    end

    local dh = SingleDataSet()
    dh:setClass("Root", "validation")
    --dh:seperateOneVsAll(opt.class)
    local label2name, name2label, all2sub, sub2all = dh:getMapping()
    
    local correct = 0
    local total = 0
    
    while true do
        print("Iter: ", total)
        local video, label_target, _ = dh:getSingleTrimedSample(10)
        if video == nil then
            break
        end
        local scores = torch.zeros(201)
        for i = 1,  video:size(1) do
            for j = 1, 201 do
                local output = svms[j]:forward(video[i])
                scores[j] = scores[j] +  output[1]
            end
        end
        for i = 1, scores:size(1) do
            --print(scores[i])
            if scores[i] ~= scores[i] then
                --print('nan')
                scores[i] = -1000.0
            end
        end
        --print("score: ", scores)
        maxVal, maxId = torch.max(scores, 1)
        --print("GT Label:")
        --print(label_target)
        --print("Pred Label:")
        --print(maxId[1])
        --io.read()
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

--local test_set = { 10, 11, 13, 14, 25, 35, 39, 44, 45, 46, 47, 48, 49}
                
--local test_set = {50, 51, 59,  66, 76, 87, 95, 97, 105, 115, 117, 119, 124, 140, 141}
                   
--local test_set = {143, 144, 146, 150, 151, 154, 155, 156, 157, 159, 161, 167, 177}
--local test_set = { 186, 188, 191, 194, 195, 196, 197, 198, 199, 200, 201}
local test_set = {201}

if opt.mode == 0 then
    for i = 1, #test_set do
        testSingleClass(test_set[i])
    end
else
    testAllClass()
end




