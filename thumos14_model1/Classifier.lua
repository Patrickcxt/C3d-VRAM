require 'dp'
--require 'rnn'
require 'ThumosDataSet'
require 'nn'
require 'cutorch'
require 'cudnn'
-- References :
-- A. http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf
-- B. http://incompleteideas.net/sutton/williams-92.pdf


version = 12

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('--xpPath', '', 'path to a previously saved model')
cmd:option('--learningRate', 0.01, 'learning rate at t=0')
cmd:option('--minLR', 0.00001, 'minimum learning rate')
cmd:option('--saturateEpoch', 500, 'epoch at which linear decayed LR will reach minLR')
cmd:option('--momentum', 0.9, 'momentum')
cmd:option('--maxOutNorm', -1, 'max norm each layers output neuron weights')
cmd:option('--cutoffNorm', -1, 'max l2-norm of contatenation of all gradParam tensors')
cmd:option('--batchSize', 12, 'number of examples per batch')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--maxEpoch', 1000000, 'maximum number of epochs to run')
cmd:option('--uniform', 0.1, 'initialize parameters using uniform distribution between -uniform and uniform. -1 means default initialization')
cmd:option('--silent', false, 'dont print anything to stdout')

cmd:text()
local opt = cmd:parse(arg or {})
--[[
if not opt.silent then
   table.print(opt)
end
]]


--[[Model]]--
if opt.xpPath ~= '' then
     assert(paths.filep(opt.xpPath), opt.xpPath..' does not exist')
     model = torch.load(opt.xpPath)
else
    --[[
    model = nn.Sequential()
    model:add(nn.Linear(1024, 512))
    model:add(nn.ReLU())
    model:add(nn.Linear(512, 100))
    model:add(nn.ReLU())
    model:add(nn.Linear(100, 20))
    ]]
    model = nn.Sequential()
    model:add(nn.Linear(16000, 1024))
    --model:add(nn.ReLU())
    --model:add(nn.Dropout(0.5))
    model:add(nn.Linear(1024, 100))
    --model:add(nn.ReLU())
    --model:add(nn.Dropout(0.5))
    model:add(nn.Linear(100, 20))


   if opt.uniform > 0 then
      for k,param in ipairs(model:parameters()) do
         param:uniform(-opt.uniform, opt.uniform)
      end
   end
end

function train()
    print(model)
    model:cuda()
    model:training()
    loss = nn.CrossEntropyCriterion()
    local dh = ThumosDataSet()

    local start_time  = sys.clock()
    for iter = 1, opt.maxEpoch do
        -- forget ?
        local input, label_targets = dh:getClassifierBatch(opt.batchSize)
        input = input:cuda()
        --label_targets = label_targets:cuda()
        local output = model:forward(input)
        output = output:double()
        local err = loss:forward(output, label_targets)
        local gradOutput = loss:backward(output, label_targets)
        gradOutput = gradOutput:cuda()
        local gradInput = model:backward(input, gradOutput)

        if (iter % 10 == 0) then
            print("iter: ", iter)
            print("Loss: ", err) 
            local duration = sys.clock() - start_time
            print("Elapsed Time: ", duration)
            start_time  = sys.clock()
            print('\n')
        end

        model:updateGradParameters(opt.momentum) -- affects gradParams
        model:updateParameters(opt.learningRate) -- affect params
        model:maxParamNorm(opt.maxOutNorm) -- affects params
        model:zeroGradParameters() -- afects gradParams

        if (iter % 2000 == 0) then
            opt.decayFactor = (opt.minLR - opt.learningRate)/opt.saturateEpoch
            opt.learningRate = opt.learningRate + opt.decayFactor
            opt.learningRate = math.max(opt.minLR, opt.learningRate)
            if not opt.silent then
                print('\n================================================')
                print("learningRate", opt.learningRate)
                print('================================================\n')
            end
        end

        if (iter % 10000 == 0) then

            -- save model
            local save_name = "./softmax3/trained_classifier_" .. tostring(iter) .. ".t7"
            torch.save(save_name, model, 'binary')
            print('\n================================================')
            print('Model ' .. save_name .. ' saved!')
            print('================================================\n')
        end
    end

    dh:destory()

end

train()

