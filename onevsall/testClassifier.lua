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
    --trained_model = torch.load('./all/classifier2/trained_classifier_300000.t7')
    trained_model = torch.load('./sports/classifier2/trained_classifier_600000.t7')
end

function demo()
    trained_model:evaluate()
    softmax = nn.SoftMax()
    local dh = SingleDataSet()
    dh:setClass("Root", "validation")

    local correct = 0
    local total = 0
    while true do
        print("Iter: ", total)
        local video, label_target, _ = dh:getSingleTrimedSample(3)
        if video == nil then
            break
        end
        local output = trained_model:forward(video)
        local pred = softmax:forward(output)
        local maxVal, maxId = torch.max(pred, 1)
        print("GT Label:")
        print(label_target)
        print("pred Label:")
        print(maxId, maxVal)
        --io.read()
        print("==============================")
        total = total + 1
        if label_target == maxId[1] then
            correct  = correct + 1
        end
    end
    print("total: " , total)
    print("correct: " , correct)
    print("Correct Ratio: ", correct / total)
end

function demo2()
    --[[ for classifier2: only use one c3d per segments ]]
    trained_model:evaluate()
    softmax = nn.SoftMax()
    local dh = SingleDataSet()
    dh:setClass("Playing sports", "validation")

    local correct = 0
    local total = 0
    while true do
        print("Iter: ", total)
        local video, label_target, _ = dh:getSingleTrimedSample()
        if video == nil then
            break
        end
        local output = trained_model:forward(video)
        local pred = softmax:forward(output)
        local maxVal, maxId = torch.max(pred, 1)
        print("GT Label:")
        print(label_target)
        print("pred Label:")
        print(maxId, maxVal)
        --io.read()
        print("==============================")
        total = total + 1
        if label_target == maxId[1] then
            correct  = correct + 1
        end
    end
    print("total: " , total)
    print("correct: " , correct)
    print("Correct Ratio: ", correct / total)
end

if opt.mode == 0 then
    demo()
else
    demo2()
end




