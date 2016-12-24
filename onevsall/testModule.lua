require 'dp'
require 'rnn'
require 'VolumetricGlimpse'
require 'DetReward'
require 'DetLossCriterion'
--require 'VideoDataHandler'
require 'MyRecurrentAttention'
require 'MyConstant'
require 'nngraph'
require 'SingleDataSet'


-- test SingleDataSet

local sh = SingleDataSet()
sh:setClass("Playing sports")

--[[
while true do
    input, label = sh:getClassifierBatch(18)
    io.read()
end
]]

-- test gModule
--[=[
agent = torch.load('./model/trained_agent_50000.t7')
gmodule = agent:get(1)

for indexNode, node in ipairs(gmodule.forwardnodes) do
    if node.data.module then
        print(node.data.module)
    end
end
rnn = gmodule:get(2)
print(rnn:get(1))
]=]




-- test VideoDataHandler
--[[
local dh = VideoDataHandler()
num1, videoList1 = dh:getTrainSet("Household Activities")
num2, videoList2 = dh:getTrainSet("Personal Care")
num3, videoList3 = dh:getTrainSet("Socializing, Relaxing, and Leisure")
num4, videoList4 = dh:getTrainSet("Sports, Exercise, and Recreation")
num5, videoList5 = dh:getTrainSet("Eating and drinking Activities")
]]









-- test DetLossCriterion Module
--[[
input = torch.Tensor({{2, 4}, {3, 8}})
target = {torch.Tensor({{1, 3}, {4, 5}}), torch.Tensor({{2, 10}})}
detreward = nn.DetLossCriterion()
output = detreward:forward(input, target)
print(output)
]]


-- test DetReward Module
--[[
input = {torch.Tensor({{2, 4}, {3, 8}}), torch.ones(3, 1)}
target = {torch.Tensor({{1, 3}, {4, 5}}), torch.Tensor({{2, 10}})}
detreward = nn.DetReward(nil, 1, nil)
output = detreward:forward(input, target)
print(output)
]]


-- test VolumetricGlimpse module
--[[
video = {}
for i = 1, 5 do
    table.insert(video, torch.Tensor(3, 32+i, 28, 28))
end
print(video)
location = torch.Tensor(20, 3)
input = {video, location}
model = nn.Sequential()
model:add(nn.VolumetricGlimpse({8, 8}, 16, 3, 2))
output = model:forward(input)
print(#output)
]]
