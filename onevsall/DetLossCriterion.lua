require 'utils'
require 'LocationSingleton'
local DetLossCriterion, parent = torch.class("nn.DetLossCriterion", "nn.Criterion")

function DetLossCriterion:__init(cls_criterion, reg_criterion)
    parent.__init(self)
    self.cls_criterion = cls_criterion or nn.MarginCriterion()
    self.reg_criterion = reg_criterion or nn.SmoothL1Criterion()
    self.cls_target, self.reg_target = {}, {}
    self.gradInput = {}
end

function DetLossCriterion:updateOutput(inputTable, target)
    local locationController = LocationSingleton:Instance()
    local locations = locationController:getLocation()

    local cls_target, reg_target = unpack(target)

    self.rho = #inputTable
    for i = 1, self.rho do
        self.cls_target[i], self.reg_target[i] = utils.get_gts(locations[i], cls_target, reg_target)
    end
    --[[
    for i = 1, self.rho do
        print(inputTable[i][2][1][1], inputTable[i][2][1][2])
        print(self.reg_target[i][1][1], self.reg_target[i][1][2])
        print('>>>>>')
    end
    ]]
    cls_err, reg_err = 0.0, 0.0
    for i = 1, self.rho do
        local cls_pred, reg_pred = unpack(inputTable[i])

        cls_err = cls_err + self.cls_criterion:updateOutput(cls_pred, self.cls_target[i])
        reg_err = reg_err + self.reg_criterion:updateOutput(reg_pred, self.reg_target[i])
    end
    cls_err, reg_err = cls_err/self.rho, reg_err/self.rho
    --print("class err: ", cls_err)
    --print("reg err: ", reg_err)
    self.output = cls_err + reg_err
    --print('Total err: ', self.output)
    return self.output
end

function DetLossCriterion:updateGradInput(inputTable, target)
    for i = 1, self.rho do
        local cls_pred, reg_pred = unpack(inputTable[i])
        self.gradInput[i] = {torch.Tensor(), torch.Tensor()}
        self.gradInput[i][1] = self.cls_criterion:updateGradInput(cls_pred, self.cls_target[i])
        self.gradInput[i][2] = self.reg_criterion:updateGradInput(reg_pred, self.reg_target[i])

    end
    return self.gradInput
end

