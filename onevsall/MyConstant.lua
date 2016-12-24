------------------------------------------------------------------------
--[[ Constant ]]--
-- Outputs a constant value given an input.
-- If nInputDim is specified, uses the input to determine the size of 
-- the batch. The value is then replicated over the batch.
-- You can use this with nn.ConcatTable() to append constant inputs to
-- an input : nn.ConcatTable():add(nn.Constant(v)):add(nn.Identity()) .
------------------------------------------------------------------------
local MyConstant, parent = torch.class("nn.MyConstant", "nn.Module")

function MyConstant:__init(value, nInputDim)
   self.value = value
   if torch.type(self.value) == 'number' then
      self.value = torch.Tensor{self.value}
   end
   assert(torch.isTensor(self.value), "Expecting number or tensor at arg 1")
   self.nInputDim = nInputDim
   parent.__init(self)
end

function MyConstant:updateOutput(input)
   if self.nInputDim and input[1][1]:dim() > self.nInputDim then
      local vsize = self.value:size():totable()
      self.output:resize(input[1][1]:size(1), table.unpack(vsize))
      local value = self.value:view(1, table.unpack(vsize))
      self.output:copy(value:expand(self.output:size())) 
   else
      self.output:resize(self.value:size()):copy(self.value)
   end
   return self.output
end

function MyConstant:updateGradInput(input, gradOutput)
   self.gradInput = {}
   for i = 1, #input do
       self.gradInput[i] = {torch.Tensor(), torch.Tensor()}
       self.gradInput[i][1]:resizeAs(input[i][1]):zero()
       self.gradInput[i][2]:resizeAs(input[i][2]):zero()
   end
   return self.gradInput
end
