require 'dp'
require 'rnn'
require 'VolumetricGlimpse'
require 'DetReward'
require 'DetLossCriterion'
require 'SingleDataSet'
require 'MyRecurrentAttention'
require 'nn'
require 'nngraph'
require 'MyConstant'
-- References :
-- A. http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf
-- B. http://incompleteideas.net/sutton/williams-92.pdf


version = 12

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Recurrent Model for Visual Attention')
cmd:text('Options:')
cmd:option('--xpPath', '', 'path to a previously saved model')
cmd:option('--learningRate', 0.0005, 'learning rate at t=0')
cmd:option('--minLR', 0.00001, 'minimum learning rate')
cmd:option('--saturateEpoch', 500, 'epoch at which linear decayed LR will reach minLR')
cmd:option('--momentum', 0.9, 'momentum')
cmd:option('--maxOutNorm', -1, 'max norm each layers output neuron weights')
cmd:option('--cutoffNorm', -1, 'max l2-norm of contatenation of all gradParam tensors')
cmd:option('--batchSize', 16, 'number of examples per batch')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--maxEpoch', 1000, 'maximum number of epochs to run')
cmd:option('--maxTries', 100, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--transfer', 'ReLU', 'activation function')
cmd:option('--uniform', 0.1, 'initialize parameters using uniform distribution between -uniform and uniform. -1 means default initialization')
cmd:option('--progress', false, 'print progress bar')
cmd:option('--silent', false, 'dont print anything to stdout')

--[[ reinforce ]]--
cmd:option('--rewardScale', 1, "scale of positive reward (negative is 0)")
cmd:option('--locatorStd', 0.11, 'stdev of gaussian location sampler (between 0 and 1) (low values may cause NaNs)')
cmd:option('--stochastic', false, 'Reinforce modules forward inputs stochastically during evaluation')

--[[ glimpse layer ]]--
cmd:option('--glimpseHiddenSize', 128, 'size of glimpse hidden layer')
cmd:option('--glimpsePatchSize', 8, 'size of glimpse patch at highest res (height = width)')
cmd:option('--glimpseScale', 2, 'scale of successive patches w.r.t. original input image')
cmd:option('--glimpseDepth', 3, 'number of concatenated downscaled patches')
cmd:option('--locatorHiddenSize', 128, 'size of locator hidden layer')
cmd:option('--imageHiddenSize', 256, 'size of hidden layer combining glimpse and locator hiddens')

--[[ recurrent layer ]]--
cmd:option('--rho', 7, 'back-propagate through time (BPTT) for rho time-steps')
cmd:option('--hiddenSize', 256, 'number of hidden units used in Simple RNN.')
cmd:option('--FastLSTM', true, 'use LSTM instead of linear layer')


cmd:option('--label', 1, 'class label to train')


cmd:text()
local opt = cmd:parse(arg or {})
if not opt.silent then
   table.print(opt)
end


--[[Model]]--
if opt.xpPath ~= '' then
     assert(paths.filep(opt.xpPath), opt.xpPath..' does not exist')
     agent = torch.load(opt.xpPath)
else

   -- glimpse network (rnn input layer)
   locationSensor = nn.Sequential()
   locationSensor:add(nn.SelectTable(2))
   locationSensor:add(nn.Linear(1, opt.locatorHiddenSize))  -- mark (loc)
   locationSensor:add(nn[opt.transfer]())

   glimpseSensor = nn.Sequential()
   glimpseSensor:add(nn.SelectTable(1))
   glimpseSensor:add(nn.Linear(500, opt.glimpseHiddenSize))
   glimpseSensor:add(nn[opt.transfer]())

   glimpse = nn.Sequential()
   glimpse:add(nn.ConcatTable():add(locationSensor):add(glimpseSensor))
   glimpse:add(nn.JoinTable(1,1))
   glimpse:add(nn.Linear(opt.glimpseHiddenSize+opt.locatorHiddenSize, opt.imageHiddenSize))
   glimpse:add(nn[opt.transfer]())
   glimpse:add(nn.Linear(opt.imageHiddenSize, opt.hiddenSize))

   -- rnn recurrent layer
   if opt.FastLSTM then
     recurrent = nn.FastLSTM(opt.hiddenSize, opt.hiddenSize)
   else
     recurrent = nn.Linear(opt.hiddenSize, opt.hiddenSize)
   end


   -- recurrent neural network
   rnn = nn.Recurrent(opt.hiddenSize, glimpse, recurrent, nn[opt.transfer](), 99999)

   -- actions (locator)
   locator = nn.Sequential()
   locator:add(nn.Linear(opt.hiddenSize, 1)) -- mark
   locator:add(nn.HardTanh()) -- bounds mean between -1 and 1
   locator:add(nn.ReinforceNormal(2*opt.locatorStd, opt.stochastic)) -- sample from normal, uses REINFORCE learning rule -- mark
   assert(locator:get(3).stochastic == opt.stochastic, "Please update the dpnn package : luarocks install dpnn")
   locator:add(nn.HardTanh()) -- bounds sample betwhen -1 and 1

   attention = nn.MyRecurrentAttention(rnn, locator, opt.rho, {opt.hiddenSize})


   agent = nn.Sequential()
   agent:add(attention)
   -- model is a reinforcement learning agent
   --[[
   finput = nn.Identity()()
   finalout = nn.Sequential()
   finalout:add(attention)
   finalout:add(nn.SelectTable(-1))
   node = finalout(finput) 
   ]]

   classifier = nn.Sequential()
   classifier:add(nn.Linear(opt.hiddenSize, 128))
   classifier:add(nn.Linear(128, 1))
   
   regression = nn.Sequential()
   regression:add(nn.Linear(opt.hiddenSize, 128))
   regression:add(nn.Linear(128, 2))

   twoStreams = {}
   twoStreams[1] = nn.ConcatTable():add(classifier):add(regression)
   prl = nn.ParallelTable()
   prl:add(twoStreams[1])
   for i = 2, opt.rho do
       twoStreams[i] = twoStreams[1]:clone('weight', 'bias', 'gradWeight', 'gradBias')
       prl:add(twoStreams[i])
   end

   agent:add(prl)


   -- classifier :

   -- add the baseline reward predictor
   seq = nn.Sequential()
   seq:add(nn.MyConstant(1,1))
   seq:add(nn.Add(1))
   concat = nn.ConcatTable():add(nn.Identity()):add(seq)
   concat2 = nn.ConcatTable():add(nn.Identity()):add(concat)

   -- output will be : {{classpred, segpred}, {{classpred, segpred}, basereward}}
   agent:add(concat2)

   if opt.uniform > 0 then
      for k,param in ipairs(agent:parameters()) do
         param:uniform(-opt.uniform, opt.uniform)
      end
   end
end

function train()
    print(agent)
    agent:training()
    loss = nn.ParallelCriterion(true)
        :add(nn.DetLossCriterion())  -- BACKPROP
        :add(nn.DetReward(agent, opt.rewardScale)) -- REINFORCE
    local dh = SingleDataSet()

    dh:setClass('Playing sports', 'training')
    dh:separateOneVsAll(opt.label)

    for iter = 1, opt.maxEpoch do
        local start_time  = sys.clock()
        -- forget ?
        local input, label_targets, segment_targets = dh:getNextBatch(opt.batchSize)

        local output = agent:forward(input)
        
        local err = loss:forward(output, {label_targets, segment_targets})
        local gradOutput = loss:backward(output, {label_targets, segment_targets})
        local gradInput = agent:backward(input, gradOutput)

        if (iter % 10 == 0) then
            print("iter: ", iter)
            print("Loss: ", err) 
            local duration = sys.clock() - start_time
            print("Elapsed Time: ", duration)
            print('\n')
        end

        agent:updateGradParameters(opt.momentum) -- affects gradParams
        agent:updateParameters(opt.learningRate) -- affect params
        agent:maxParamNorm(opt.maxOutNorm) -- affects params
        agent:zeroGradParameters() -- afects gradParams

        if (iter % 100 == 0) then
            opt.decayFactor = (opt.minLR - opt.learningRate)/opt.saturateEpoch
            opt.learningRate = opt.learningRate + opt.decayFactor
            opt.learningRate = math.max(opt.minLR, opt.learningRate)
            if not opt.silent then
                print('\n================================================')
                print("learningRate", opt.learningRate)
                print('================================================\n')
            end
        end

        if (iter % 40 == 0) then

            -- save model
            local save_name = "./model/agent_"  .. tostring(opt.label) .. '_'.. tostring(iter) .. ".t7"
            torch.save(save_name, agent, 'binary')
            print('\n================================================')
            print('Model ' .. save_name .. ' saved!')
            print('================================================\n')
        end
        --io.read()
    end

    dh:destroy()

end

train()

