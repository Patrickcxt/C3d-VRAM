------------------------------------------------------------------------
--[[ MyMyRecurrentAttention ]]-- 
-- Ref. A. http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf
-- B. http://incompleteideas.net/sutton/williams-92.pdf
-- module which takes an RNN as argument with other 
-- hyper-parameters such as the maximum number of steps, 
-- action (actions sampling module like ReinforceNormal) and 
------------------------------------------------------------------------
local MyRecurrentAttention, parent = torch.class("nn.MyRecurrentAttention", "nn.AbstractSequencer")

require 'utils'
require 'LocationSingleton'

function MyRecurrentAttention:__init(rnn, action, nStep, hiddenSize)
   parent.__init(self)
   assert(torch.isTypeOf(action, 'nn.Module'))
   assert(torch.type(nStep) == 'number')
   assert(torch.type(hiddenSize) == 'table')
   assert(torch.type(hiddenSize[1]) == 'number', "Does not support table hidden layers" )
   
   self.rnn = rnn
   -- we can decorate the module with a Recursor to make it AbstractRecurrent
   self.rnn = (not torch.isTypeOf(rnn, 'nn.AbstractRecurrent')) and nn.Recursor(rnn) or rnn
   
   -- samples an x,y actions for each example
   self.action =  (not torch.isTypeOf(action, 'nn.AbstractRecurrent')) and nn.Recursor(action) or action 
   self.hiddenSize = hiddenSize
   self.nStep = nStep
   
   self.modules = {self.rnn, self.action}
   
   self.output = {} -- rnn output
   self.actions = {} -- action output
   
   self.forwardActions = false
   
   self.gradHidden = {}
   
   self.clips = {} -- video clips
   self.time = 16
end

function MyRecurrentAttention:updateOutput(input)
   self.rnn:forget()
   self.action:forget()

   for step=1,self.nStep do
      
      if step == 1 then
         -- sample an initial starting actions by forwarding zeros through the action
         self._initInput = self._initInput or torch.Tensor()
         self._initInput:resize(#input, table.unpack(self.hiddenSize)):zero()
         self.actions[1] = self.action:updateOutput(self._initInput)
      else
         -- sample actions from previous hidden activation (rnn output)
         self.actions[step] = self.action:updateOutput(self.output[step-1])
      end
      
      -- fix: trim videos
      self.actions[step] = (self.actions[step] + 1.0) / 2.0

      locationController = LocationSingleton:Instance()
      locationController:setLocation(self.actions[step], step)
      self.clips[step] = self:trim_video(input, self.actions[step])
      -- rnn handles the recurrence internally
      local output = self.rnn:updateOutput{self.clips[step], self.actions[step]}
      self.output[step] = self.forwardActions and {output, self.actions[step]} or output
   end
   return self.output
end

function MyRecurrentAttention:updateGradInput(input, gradOutput)
   assert(self.rnn.step - 1 == self.nStep, "inconsistent rnn steps")
   assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
   assert(#gradOutput == self.nStep, "gradOutput should have nStep elements")
    
   -- back-propagate through time (BPTT)
   for step=self.nStep,1,-1 do
      -- 1. backward through the action layer
      local gradOutput_, gradAction_ = gradOutput[step]
      if self.forwardActions then
         gradOutput_, gradAction_ = unpack(gradOutput[step])
      else
         -- Note : gradOutput is ignored by REINFORCE modules so we give a zero Tensor instead
         self._gradAction = self._gradAction or self.action.output.new()
         if not self._gradAction:isSameSizeAs(self.action.output) then
            self._gradAction:resizeAs(self.action.output):zero()
         end
         gradAction_ = self._gradAction
      end
      
      if step == self.nStep then
         self.gradHidden[step] = nn.rnn.recursiveCopy(self.gradHidden[step], gradOutput_)
      else
         -- gradHidden = gradOutput + gradAction
         nn.rnn.recursiveAdd(self.gradHidden[step], gradOutput_)
      end
      
      if step == 1 then
         -- backward through initial starting actions
         self.action:updateGradInput(self._initInput, gradAction_)
      else
         local gradAction = self.action:updateGradInput(self.output[step-1], gradAction_)
         self.gradHidden[step-1] = nn.rnn.recursiveCopy(self.gradHidden[step-1], gradAction)
      end
      
      -- 2. backward through the rnn layer
      local gradInput = self.rnn:updateGradInput({self.clips[step], self.actions[step]}, self.gradHidden[step])[1]
      if step == self.nStep then
         self.gradInput:resizeAs(gradInput):copy(gradInput)
      else
         self.gradInput:add(gradInput)
      end
   end

   return self.gradInput
end

function MyRecurrentAttention:accGradParameters(input, gradOutput, scale)
   assert(self.rnn.step - 1 == self.nStep, "inconsistent rnn steps")
   assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
   assert(#gradOutput == self.nStep, "gradOutput should have nStep elements")
   
   -- back-propagate through time (BPTT)
   for step=self.nStep,1,-1 do
      -- 1. backward through the action layer
      local gradAction_ = self.forwardActions and gradOutput[step][2] or self._gradAction
            
      if step == 1 then
         -- backward through initial starting actions
         self.action:accGradParameters(self._initInput, gradAction_, scale)
      else
         self.action:accGradParameters(self.output[step-1], gradAction_, scale)
      end
      
      -- 2. backward through the rnn layer
      self.rnn:accGradParameters({self.clips[step], self.actions[step]}, self.gradHidden[step], scale)
   end
end

function MyRecurrentAttention:accUpdateGradParameters(input, gradOutput, lr)
   assert(self.rnn.step - 1 == self.nStep, "inconsistent rnn steps")
   assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
   assert(#gradOutput == self.nStep, "gradOutput should have nStep elements")
    
   -- backward through the action layers
   for step=self.nStep,1,-1 do
      -- 1. backward through the action layer
      local gradAction_ = self.forwardActions and gradOutput[step][2] or self._gradAction
      
      if step == 1 then
         -- backward through initial starting actions
         self.action:accUpdateGradParameters(self._initInput, gradAction_, lr)
      else
         -- Note : gradOutput is ignored by REINFORCE modules so we give action.output as a dummy variable
         self.action:accUpdateGradParameters(self.output[step-1], gradAction_, lr)
      end
      
      -- 2. backward through the rnn layer
      self.rnn:accUpdateGradParameters({self.clips[step], self.actions[step]}, self.gradHidden[step], lr)
   end
end

function MyRecurrentAttention:type(type)
   self._input = nil
   self._actions = nil
   self._crop = nil
   self._pad = nil
   self._byte = nil
   return parent.type(self, type)
end

function MyRecurrentAttention:__tostring__()
   local tab = '  '
   local line = '\n'
   local ext = '  |    '
   local extlast = '       '
   local last = '   ... -> '
   local str = torch.type(self)
   str = str .. ' {'
   str = str .. line .. tab .. 'action : ' .. tostring(self.action):gsub(line, line .. tab .. ext)
   str = str .. line .. tab .. 'rnn     : ' .. tostring(self.rnn):gsub(line, line .. tab .. ext)
   str = str .. line .. '}'
   return str
end

function MyRecurrentAttention:trim_video(input, loc)
    local clips = torch.Tensor(#input, 16000)
    for i = 1, #input do
        local l = loc[i]:select(1, 1)
        local frameIdx = math.min(input[i]:size(1), math.max(1, math.round(l * input[i]:size(1))))
        clips[i] = input[i][frameIdx]
        local num_idts = clips[i][{{1, 4000}}]:sum()
        clips[i]:div(num_idts)
    end
    return clips
end

--[[
function MyRecurrentAttention:setMode(mode)
    self.mode = mode
    if self.mode == 'training' then
        print('Mode: training')
        self.annationFile = hdf5.open('/home/amax/cxt/data/THUMOS2014/validation/video_infos.h5', 'r')
        self.path = '/home/amax/cxt/data/THUMOS2014/validation/TH14_validation_features/'
    else
        print('Mode: test')
        self.annationFile = hdf5.open('/home/amax/cxt/data/THUMOS2014/test/video_infos.h5', 'r')
        self.path = '/home/amax/cxt/data/THUMOS2014/test/TH14_test_features/'
    end
end
function MyRecurrentAttention:trim_video(input, loc)
    local clips = torch.Tensor(#input, 16000)
    for i = 1, #input do
        local l = loc[i]:select(1, 1)
        local infos = self.annationFile:read(input[i]):all()
        local duration, frame_num =  infos[1], infos[3]
        local frameIdx = math.max(1, math.floor(l * frame_num))
        local fn = self.path .. input[i] .. '.txt'
        clips[i] = self:get_idt(fn, frameIdx-15, frameIdx+16)
    end
    return clips
end
]]

--[[
function MyRecurrentAttention:get_idt(video_path, st, ed)
    --print(video_path, st, ed)
    local f = io.open(video_path, 'r')
    local iDTs = {}
    for line in f:lines() do
        local infos = utils.split(line, '\t') 
        local frame = tonumber(infos[1])
        if frame > ed then break end
        if frame >= st and frame <= ed then
            local feature_vector = torch.Tensor({infos[4], infos[5], infos[6], infos[7]})
            feature_vector:add(1)
            table.insert(iDTs, feature_vector)
        end
    end
    local vector = self:BoFPooling(iDTs)
    return vector
end

function MyRecurrentAttention:BoFPooling(iDTs)
    local Traj, Hog, Hof, Mbh = torch.zeros(4000), torch.zeros(4000), torch.zeros(4000), torch.zeros(4000)
    for i = 1, #iDTs do
        local traj, hog, hof, mbh = iDTs[i][1],iDTs[i][2], iDTs[i][3], iDTs[i][4]
        Traj[traj] = Traj[traj] + 1
        Hog[hog] = Hog[hog] + 1
        Hof[hof] = Hof[hof] + 1

        Mbh[mbh] = Mbh[mbh] + 1
    end
    local feat = Traj:cat(Hog):cat(Hof):cat(Mbh)
    feat:div(#iDTs)
    return feat
end
]]
