require 'utils'
local DetLossCriterion, parent = torch.class("nn.DetLossCriterion", "nn.Criterion")

function DetLossCriterion:__init(criterion)
    parent.__init(self)
    self.criterion = nn.SequencerCriterion(nn.SmoothL1Criterion())
    self.gradInput = torch.Tensor()
    self.reg_target = {}
end

function DetLossCriterion:updateOutput(inputTable, target)
    for i = 1, #inputTable do
        self.reg_target[i] = utils.get_gts(inputTable[i], target)
    end
    print(inputTable[7])
    print(self.reg_target[7])
    local reg_err  = self.criterion:updateOutput(inputTable, self.reg_target)
    print("reg err: ", reg_err)
    self.output = reg_err
    return self.output
end

function DetLossCriterion:updateGradInput(inputTable, target)
    self.gradInput = self.criterion:updateGradInput(inputTable, self.reg_target)
    return self.gradInput
end

