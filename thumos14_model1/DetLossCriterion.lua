require 'utils'
require 'LocationSingleton'
local DetLossCriterion, parent = torch.class("nn.DetLossCriterion", "nn.Criterion")

function DetLossCriterion:__init(criterion)
    parent.__init(self)
    self.criterion = nn.SequencerCriterion(nn.SmoothL1Criterion())
    self.gradInput = torch.Tensor()
    self.reg_target = {}
end

function DetLossCriterion:updateOutput(inputTable, target)
    local locationController = LocationSingleton:Instance()
    local locations = locationController:getLocation()

    for i = 1, #inputTable do
        self.reg_target[i] = utils.get_gts(locations[i], target)
    end
    for i = 1, 14 do
        print(inputTable[i][1][1], inputTable[i][1][2])
    end
    print('\n')
    for i = 1, 14 do
        print(self.reg_target[i][1][1], self.reg_target[i][1][2])
    end

    local reg_err  = self.criterion:updateOutput(inputTable, self.reg_target)
    print("reg err: ", reg_err)
    self.output = reg_err
    return self.output
end

function DetLossCriterion:updateGradInput(inputTable, target)
    self.gradInput = self.criterion:updateGradInput(inputTable, self.reg_target)
    return self.gradInput
end

