
LocationSingleton = {}

function LocationSingleton:new(obj)
    obj = obj or {}
    setmetatable(obj, self)
    self.__index = self
    self.locs = torch.Tensor(7, 48)
    return obj
end

function LocationSingleton:Instance()
    if self.instance == nil then
        self.instance = self:new()
    end
    return self.instance
end

function LocationSingleton:setLocation(locs, step)
    self.locs[step]:resizeAs(locs):copy(locs:view(-1))
end

function LocationSingleton:getLocation()
    return self.locs
end

return LocationSingleton
