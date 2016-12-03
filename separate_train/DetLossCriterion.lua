require 'utils'
local DetLossCriterion, parent = torch.class("nn.DetLossCriterion", "nn.Criterion")

function DetLossCriterion:__init(criterion)
    parent.__init(self)
    self.criterion = criterion or nn.SmoothL1Criterion()
    self.gradInput = torch.Tensor()
end

function DetLossCriterion:updateOutput(inputTable, target)
    local reg_pred = inputTable
    self.reg_target = utils.get_gts(reg_pred, target)
    local reg_err  = self.criterion:updateOutput(reg_pred, self.reg_target)
    --print("reg err: ", reg_err)
    self.output = reg_err
    return self.output
end

function DetLossCriterion:updateGradInput(inputTable, target)
    local reg_pred = inputTable
    self.gradInput = self.criterion:updateGradInput(reg_pred, self.reg_target)
    return self.gradInput
end

