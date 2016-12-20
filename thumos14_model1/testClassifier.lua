require 'nn'
require 'dp'
require 'ThumosDataSet'
require 'cutorch'
require 'cudnn'

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


cutorch.setDevice(2)
--[[Model]]--
if opt.xpPath ~= '' then
    assert(paths.filep(opt.xpPath), opt.xpPath ..' does not exist')
    trained_model = torch.load(opt.xpPath)
els 
    --trained_model = torch.load('./all/classifier2/trained_classifier_300000.t7')
    trained_model = torch.load('./softmax3/trained_classifier_100000.t7')
end

function demo()
    trained_model:evaluate()
    softmax = nn.SoftMax()
    local dh = ThumosDataSet()

    local correct = 0
    local total = 0
    while true do
        print("Iter: ", total)
        local video, label_target = dh:getSingleValSample()
        if video == nil then
            break
        end
        video = video:cuda()
        local output = trained_model:forward(video)
        output = output:double()
        local pred = softmax:forward(output)
        local maxVal, maxId = torch.max(pred, 1)
        print("GT Label:")
        print(label_target)
        print("pred Label:")
        print(maxId[1], maxVal[1])
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

demo()



