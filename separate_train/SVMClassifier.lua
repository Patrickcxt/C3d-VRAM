require 'dp'
require 'rnn'
require 'SingleDataSet'
require 'nn'
-- References :
-- A. http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf
-- B. http://incompleteideas.net/sutton/williams-92.pdf


version = 12

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('--learningRate', 0.01, 'learning rate at t=0')
cmd:option('--minLR', 0.00001, 'minimum learning rate')
cmd:option('--saturateEpoch', 800, 'epoch at which linear decayed LR will reach minLR')
cmd:option('--momentum', 0.9, 'momentum')
cmd:option('--maxOutNorm', -1, 'max norm each layers output neuron weights')
cmd:option('--cutoffNorm', -1, 'max l2-norm of contatenation of all gradParam tensors')
cmd:option('--batchSize', 18, 'number of examples per batch')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--maxEpoch', 800, 'maximum number of epochs to run')
cmd:option('--uniform', 0.1, 'initialize parameters using uniform distribution between -uniform and uniform. -1 means default initialization')
cmd:option('--silent', false, 'dont print anything to stdout')
cmd:option('--class', 1, 'class to train')

cmd:text()
local opt = cmd:parse(arg or {})
if not opt.silent then
   table.print(opt)
end



function train(clsLabel)
    print("Begin to train the svm model:", clsLabel)
    --[[Model]]--

    opt.learningRate = 0.01
    --[[
    local model = nn.Sequential()
    model:add(nn.Linear(500, 256))
    model:add(nn.ReLU())
    model:add(nn.Linear(256, 128))
    model:add(nn.ReLU())
    model:add(nn.Linear(128, 1))
    ]]
    local model = torch.load('./all/svms/201_svm_800.t7')

    if opt.uniform > 0 then
        for k,param in ipairs(model:parameters()) do
            param:uniform(-opt.uniform, opt.uniform)
        end
    end
    model:training()
    local loss = nn.MarginCriterion()

    local dh = SingleDataSet()
    dh:setClass("Root", "train")
    dh:seperateOneVsAll(clsLabel)   -- seperate all timed video into pos and neg part

    local total_loss = 0.0
    for iter = 1, opt.maxEpoch do
        local start_time  = sys.clock()
        -- forget ?
        local input, label_targets = dh:getSVMClassifierBatch(opt.batchSize)
        local output = model:forward(input)
        local err = loss:forward(output, label_targets)
        total_loss = total_loss + err
        local gradOutput = loss:backward(output, label_targets)
        local gradInput = model:backward(input, gradOutput)

        if (iter % 50 == 0) then
            print("iter: ", iter)
            print("Loss: ", total_loss / 50) 
            total_loss = 0.0
            local duration = sys.clock() - start_time
            print("Elapsed Time: ", duration)
            print('\n')
        end

        model:updateGradParameters(opt.momentum) -- affects gradParams
        model:updateParameters(opt.learningRate) -- affect params
        model:maxParamNorm(opt.maxOutNorm) -- affects params
        model:zeroGradParameters() -- afects gradParams

        if (iter % 500 == 0) then
            opt.decayFactor = (opt.minLR - opt.learningRate)/opt.saturateEpoch
            opt.learningRate = opt.learningRate + opt.decayFactor
            opt.learningRate = math.max(opt.minLR, opt.learningRate)
            --[[
            if not opt.silent then
                print('\n================================================')
                print("learningRate", opt.learningRate)
                print('================================================\n')
            end
            ]]
        end

        if (iter % 200 == 0) then

            -- save model
            local save_name = "./all/svms/" .. tostring(clsLabel) .. "_svm_" .. tostring(iter) .. ".t7"
            torch.save(save_name, model, 'binary')
            print('\n================================================')
            print('Model ' .. save_name .. ' saved!')
            print('================================================\n')
        end
    end

    dh:destroy()
    print("Done..")
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n')

end

--[[
local train_set = {6, 7, 9, 10, 11, 13, 14, 25, 35, 39, 44, 45, 46, 47, 48, 49, 50, 51, 59, 
                   66, 76, 87, 95, 97, 105, 115, 117, 119, 124, 140, 141, 143, 144, 146, 150,
                   151, 154, 155, 156, 157, 159, 161, 167, 177, 186, 188, 191, 194, 195, 196,
                   197, 198, 199, 200, 201}
]]
local train_set = { 10, 11, 13, 14, 25, 35, 39, 44, 45, 46, 47, 48, 49, 50, 51, 59, 
                   66, 76, 87, 95, 97, 105, 115, 117, 119, 124, 140, 141, 143, 144, 146, 150,
                   151, 154, 155, 156, 157, 159, 161, 167, 177, 186, 188, 191, 194, 195, 196,
                   197, 198, 199, 200, 201}
    
train(opt.class)

