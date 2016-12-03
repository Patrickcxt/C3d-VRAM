require 'dp' 
require 'rnn' 
require 'VolumetricGlimpse' 
require 'DetReward' 
require 'DetLossCriterion' 
require 'VideoDataHandler' 
require 'MyRecurrentAttention' 
require 'nngraph' 
require 'MyConstant' 
require 'SingleDataSet'
                        
local cjson = require 'cjson'
--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Test a classifier Model')
cmd:text('Example:')
cmd:text('$> th test.lua > results.txt')
cmd:text('Options:')
cmd:option('--rnnPath', '', 'path to a previously saved model')
cmd:option('--agentPath', '', 'path to a previously saved model')
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
if opt.rnnPath ~= '' and opt.agentPath ~= '' then
    assert(paths.filep(opt.rnnPath), opt.rnnPath ..' does not exist')
    assert(paths.filep(opt.agentPath), opt.agentPath ..' does not exist')
    rnn = torch.load(opt.rnnPath)
    agent = torch.load(opt.agentPath)
else
    assert(false)
end


function _clip_segments(segments)
    segments[{{}, {1}}]:cmax(0)
    segments[{{}, {1}}]:cmin(1)
    segments[{{}, {2}}]:cmax(0)
    segments[{{}, {2}}]:cmin(1)
    segments[{{}, {1}}]:cmin(segments[{{}, {2}}])
    return segments
end


function demo()
    agent:evaluate()
    softmax = nn.SoftMax()
    local dh = SingleDataSet()
    local _, valSet, _, _ = dh:setClass("Root", "validation")
    local label2name, name2label, all2sub, sub2all = dh:getMapping()

    local correct = 0
    local total = 0
    local randIdx = torch.randperm(#valSet)
    for i = 1, #valSet do
        print(valSet[randIdx[i]])
        local video, _, segment_target = dh:getSingleSample(valSet[randIdx[i]])
        local rnn_output = rnn:forward({video})
        local pred = torch.Tensor(#rnn_output, 2)
        for j = 1, #rnn_output do
            local output = agent:forward(rnn_output[j])
            pred[j] = output[1]
        end
        pred = _clip_segments(pred)
        local overlap = utils.interval_overlap(segment_target, pred)
        print(pred)
        print(segment_target)
        print(overlap)
        io.read()

    end
    --[[
    print("total: " , total)
    print("correct: " , correct)
    print("Correct Ratio: ", correct / total)
    ]]
end
demo()




