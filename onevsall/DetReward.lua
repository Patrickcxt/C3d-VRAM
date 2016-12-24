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
   self.module = module -- so it can call module:reinforce(reward) 
   self.scale = scale or 1 -- scale of reward
   self.criterion = criterion or nn.MSECriterion() -- baseline criterion
   self.sizeAverage = true
   self.gradInput = {{}, torch.Tensor()}
   self.threshold = threshold or 0.65
end

function DetReward:updateOutput(inputTable, target)
   local input = inputTable[1]  -- preds
   self.steps = #input
   local batch_size = input[1][1]:size(1)
   local cls_target, reg_target= unpack(target)
   
   self._reward = torch.Tensor(batch_size):zero()

   local cls_pred, reg_pred = input[1][1], input[1][2]
   for step = 2, self.steps do
       cls_pred = torch.cat(cls_pred, input[step][1], 2)
       reg_pred = torch.cat(reg_pred, input[step][2], 2)
   end
   cls_pred:resize(batch_size, self.steps, 1)
   reg_pred:resize(batch_size, self.steps, 2)

   -- Get GT label
   local controller = LocationSingleton:Instance()
   local locations = controller:getLocation()
   local gt_label = torch.Tensor(batch_size, self.steps)
   for i = 1, self.steps do
       gt_label[{{}, {i}}] = utils.get_gts(locations[i], cls_target, reg_target)
   end
   gt_label:resize(batch_size, self.steps, 1)

   for b = 1, batch_size do
       local c_pred = torch.Tensor(self.steps, 1)
       local r_pred = torch.Tensor(self.steps, 2)
       c_pred:copy(cls_pred[b])
       r_pred:copy(reg_pred[b])
    
       for j = 1, c_pred:size(1) do
           if c_pred[j][1] >= 0.0 then c_pred[j][1] = 1
           else c_pred[j][1] = -1 end
       end

       local r1 = c_pred:eq(gt_label[b]):sum()
       --print('r1: ', r1)

       local gts = reg_target[b]
       local overlap = utils.interval_overlap(gts, r_pred)
       local max_ov, max_idx = torch.max(overlap, 1)
       --print(max_ov)
       --print(max_idx)
       local r2 = max_ov:ge(self.threshold):sum()
       --print('r2: ', r2)
       self._reward[b] = (r1 + r2 ) / 2.0
   end

   self.reward = self._reward
   self.reward = self.reward:mul(self.scale)

   -- loss = -sum(reward)
   self.output = -self.reward:sum()
   if self.sizeAverage then
      self.output = self.output/batch_size
   end
   --print("Reward: ", self.output)
   --print('\n')
   return self.output
end

function DetReward:updateGradInput(inputTable, target)
   local pred = inputTable[1]
   local baseline = inputTable[2]
   
   -- reduce variance of reward using baseline
   --[[
   print("reward")
   print(self.reward)
   ]]
   self.detReward = self.detReward or self.reward.new()
   self.detReward:resizeAs(self.reward):copy(self.reward)
   self.detReward:add(-1, baseline)    -- add -1 * baseline 
   --[[
   print("detReward")
   print(self.detReward)
   io.read()
   ]]

   if self.sizeAverage then
      self.detReward:div(pred[1][1]:size(1))
   end
   -- broadcast reward to modules
   self.module:reinforce(self.detReward)  
   
   -- zero gradInput (this criterion has no gradInput for class pred)
   for i = 1, self.steps do
       self.gradInput[1][i] = {torch.Tensor(), torch.Tensor()}
       self.gradInput[1][i][1]:resizeAs(pred[1][1]):zero()
       self.gradInput[1][i][2]:resizeAs(pred[1][2]):zero()
   end
   
   -- learn the baseline reward
   self.criterion:forward(baseline, self.reward)
   self.gradInput[2] = self.criterion:backward(baseline, self.reward)
   --self.gradInput[2] = self:fromBatch(self.gradInput[2], 1)
   return self.gradInput
end

