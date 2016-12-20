--require 'ThumosDataSet'
--dh = ThumosDataSet()
require 'gnuplot'
x = torch.linspace(-2*math.pi, 2*math.pi)
print(x)
gnuplot.plot(torch.sin(x))


--[[
model = nn.Sequential()
model:add(nn.Linear(100000, 10000))
model:add(nn.ReLU())
model:add(nn.Dropout(0.5))
model:add(nn.Linear(10000, 1000))
model:add(nn.ReLU())
model:add(nn.Dropout(0.5))
model:add(nn.Linear(1000, 100))
model:cuda()

x = torch.Tensor(100000)
x = x:cuda()

for i = 1, 100 do
    print(i)
    model:forward(x)
end
]]
