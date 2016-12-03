------------------------------------------------------------------------
--[[ DetReward ]]--
-- Variance reduced detection reinforcement criterion.
-- input : {detection prediction, baseline reward}
-- Reward is 1 for success, Reward is 0 otherwise.
-- reward = scale*(Reward - baseline) where baseline is 2nd input element
-- Note : for RNNs with R = 1 for last step in sequence, encapsulate it
-- in nn.ModuleCriterion(DetReward, nn.SelectTable(-1))
------------------------------------------------------------------------

require 'utils'
local DetReward, parent = torch.class("nn.DetReward", "nn.Criterion")

function DetReward:__init(module, scale, criterion)
   parent.__init(self)
   self.module = module -- so it can call module:reinforce(reward)  agent?!
   self.scale = scale or 1 -- scale of reward
   self.criterion = criterion or nn.MSECriterion() -- baseline criterion
   self.sizeAverage = true
   self.gradInput = {torch.Tensor(), torch.Tensor()}
   self.threshold = threshold or 0.5
end

function DetReward:updateOutput(inputTable, target)
   local reg_pred = inputTable[1]
   local batch_size = reg_pred:size(1)
   
   self.reg_reward = torch.Tensor(batch_size):zero()
   for b = 1, batch_size do
       local pred = torch.Tensor(2)
       pred:copy(reg_pred[b]):resize(1, 2)
       local gts = target[b]
       local overlap = utils.interval_overlap(gts, pred)
       local max_ov, max_idx = torch.max(overlap, 1)
       --print(gts[max_idx[1][1]])
       if max_ov[1][1] > self.threshold then
           self.reg_reward[b] = 1
       end
   end

   self.reward = self.reg_reward
   self.reward = self.reward:mul(self.scale)

   -- loss = -sum(reward)
   self.output = -self.reward:sum()
   if self.sizeAverage then
      self.output = self.output/reg_pred:size(1)
   end
   --print("Reward: ", self.output)
   return self.output
end

function DetReward:updateGradInput(inputTable, target)
   local reg_pred = inputTable[1]
   local baseline = inputTable[2]
   
   -- reduce variance of reward using baseline
   print("reward")
   print(self.reward)
   self.detReward = self.detReward or self.reward.new()
   self.detReward:resizeAs(self.reward):copy(self.reward)
   self.detReward:add(-1, baseline)    -- puts -1 + baseline in self.detReward
   print("detReward")
   print(self.detReward)
   io.read()

   if self.sizeAverage then
      self.detReward:div(reg_pred:size(1))
   end
   print(self.detReward)
   -- broadcast reward to modules
   self.module:reinforce(self.detReward)  
   
   -- zero gradInput (this criterion has no gradInput for class pred)
   self.gradInput[1]:resizeAs(reg_pred):zero()
   
   -- learn the baseline reward
   self.criterion:forward(baseline, self.reward)
   self.gradInput[2] = self.criterion:backward(baseline, self.reward)
   --self.gradInput[2] = self:fromBatch(self.gradInput[2], 1)
   return self.gradInput
end

