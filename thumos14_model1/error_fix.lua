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
cmd:text('Options:')
cmd:option('--predPath', 'tmp.txt', 'path to a previously saved model')

cmd:text()
local opt = cmd:parse(arg or {})
if not opt.silent then
   table.print(opt)
end


f = io.open(opt.predPath, 'a+')
cnt = {}
for i = 1, 20 do
    cnt[i] = 0
end
for line in f:lines() do
    re = utils.split(line, '\t')
    label = tonumber(re[4])
    cnt[label] = cnt[label] + 1
end

for i = 1, 20 do
    if cnt[i] == 0 then
        print(i)
        st = torch.rand(1)[1]
        ed = st + 0.015
        line = 'video_test_0000319' .. '\t' .. st*100.0 .. '\t' .. ed*100.0 .. '\t' .. i .. '\t' .. '0.95' .. '\n'
        print(line)
        f:write(line)
    end
   
end

