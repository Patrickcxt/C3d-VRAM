-----------------------------------------------------------------------
--[[ SingleDataSet ]]--
--Integrating some operations to deal with video data.
-----------------------------------------------------------------------
require 'ffmpeg'
require 'image'
require 'utils'
require 'lfs'
require 'hdf5'
local cjson = require 'cjson'
--local torchvid = require 'torchvid'

local SingleDataSet = torch.class("SingleDataSet")

function SingleDataSet:__init(rootPath, idxPath, jsPath) -- /path/to/your/Videos/
	self.rootPath = rootPath or '/media/chenxt/wjz1/data/Videos/activitynet/'
    self.idxPath = idxPath or '/home/chenxt/cxt/data/ActivityNet/scheme/index.txt'
    self.jsPath = jsPath or "/home/chenxt/cxt/data/ActivityNet/scheme/activity_net.v1-3.min.json"
    self.cachePath = "/home/chenxt/cxt/VRAM/cache/"
    self.c3dPath = "/home/chenxt/cxt/data/ActivityNet/features/sub_activitynet_v1-3.c3d.hdf5"
    self.mbhPath = "/ActivityNet/MBH_Videos_features_2.h5"
    --self.mbhIdxPath = "/home/chenxt/cxt/data/ActivityNet/scheme/MBH_Videos_quids.txt"
    --self.c3dPath = "/home/chenxt/cxt/data/ActivityNet/features/ds_activitynet_v1-3.c3d.hdf5" -- downsample

	self.videoList, self.cutNameList = self:create_video_list()
	self.database, self.version, self.taxonomy = self:parse_json()

	self.root, self.id2Tax = self:create_taxonomy_tree()
    self.name2label, self.label2name = self:get_classes()
	self.trainSet, self.valSet, self.testSet = self:create_dataset()
    self.c3dFile = hdf5.open(self.c3dPath, 'r')
    self.mbhFile = hdf5.open(self.mbhPath, 'r')
end

function SingleDataSet:setClass(clsName, mode)
    self.clsName = clsName
    self.subClasses, self.subTrainSet, self.subValSet, 
        self.subTestSet, self.negSet = self:getSubSet()
    print(self.subClasses)
	self.sub_label2name, self.sub_name2label, self.all2sub, self.sub2all = self:get_sub_classes()
    self.posTrimed, self.negTrimed = self:get_trimed_dataset(mode)
    --print("num of pos trimed: ", #self.posTrimed)
	self.posIdx = 1
    self.negIdx = 1
    return self.subTrainSet, self.subValSet, self.subTestSet, self.negSet
end

function SingleDataSet:seperateOneVsAll(clsLabel)
    --[=[
        For trimed video SVM Classifier
    ]=]
    local posTrimed = {}
    local negTrimed = {} 
    for i = 1, #self.posTrimed do
        local trimed = self.posTrimed[i]
        if trimed["gt_labels"] == clsLabel then
            table.insert(posTrimed, trimed)
        else
            table.insert(negTrimed, trimed)
        end
    end
    for i = 1, #self.negTrimed do
        local trimed = self.negTrimed[i]
        if trimed["gt_labels"] == clsLabel then
            table.insert(posTrimed, trimed)
        else
            table.insert(negTrimed, trimed)
        end
    end
    self.posTrimed = posTrimed
    self.negTrimed = negTrimed
    print("Class Name: ", self.sub_label2name[clsLabel])
    print("Postivie Number: ", #self.posTrimed)
    print("Negative Number: ", #self.negTrimed)
end

function SingleDataSet:seperateOneVsAllGlobal(clsLabel, mode)
    --[=[
        For global video SVM Classifier
    ]=]
    local posSet = {}
    local negSet = {} 
    local targetSet = nil
    if mode == "train" then
        targetSet = self.subTrainSet
    elseif mode == "validation" then
        targetSet = self.subValSet
    else
        targetSet = self.subTestSet
    end

    for i = 1, #targetSet do
        local name = targetSet[i]
        local vidInfo = self.database[self:cut_name(name)]    
        local label = self.all2sub[vidInfo["gt_labels"][1]]
        if label == clsLabel then
            table.insert(posSet, name)
        else
            table.insert(negSet, name)
        end
    end
    -- negSet will not be used when testing
    for i = 1, #self.negSet do
        local name = self.negSet[i]
        --local vidInfo = self.database[self:cut_name(name)]    
        local label = #self.subClasses + 1
        if label == clsLabel then
            table.insert(posSet, name)
        else
            table.insert(negSet, name)
        end
    end
    self.posSet = posSet
    self.negSet = negSet
    print("Class Name: ", self.sub_label2name[clsLabel])
    print("Postivie Number: ", #self.posSet)
    print("Negative Number: ", #self.negSet)
    return self.posSet, self.negSet
end

function SingleDataSet:getMapping()
    return self.sub_label2name, self.sub_name2label,  self.all2sub, self.sub2all
end

function SingleDataSet:getSubSet()
    if self.clsName == nil then
        return #self.label2name, self.trainSet, self.valSet
    end
    local subClasses = {}
    local trainSet = {}
    local valSet = {}
    local testSet = {}
	for id, taxon in pairs(self.id2Tax) do
        if taxon.nodeName == self.clsName then
            subClasses = self:retrieval_all_leaves(taxon)
            break
        end
    end
    self.subClasses = subClasses
    subClasses = utils.Set(subClasses)
    for i = 1, #self.cutNameList do
        local vidInfo = self.database[self.cutNameList[i]]    
        local gt_labels = vidInfo["gt_labels"]
        for j = 1, gt_labels:size(1) do
            local lname = self.label2name[gt_labels[j]]
            if subClasses[lname] == true then
                if vidInfo["subset"] == "training" then
                    table.insert(trainSet, self.videoList[i])
                elseif vidInfo["subset"] == "validation" then
                    table.insert(valSet, self.videoList[i])
                else
                    table.insert(testSet, self.videoList[i])
                end
                break
            end
        end
    end
    local negSet = self:get_negative_set(trainSet)
    return self.subClasses, trainSet, valSet, testSet, negSet
end


function SingleDataSet:get_sub_classes()
	local sub_name2label = {}
    local all2sub = {}  -- label mapping from all classes to sub classes
    local sub2all = {}  -- label mapping from sub classes to all classes
	local cnt = 1
	for i = 1, #self.subClasses do
        local name = self.subClasses[i]
        sub_name2label[name] = i
        all2sub[self.name2label[name]] = i
        sub2all[i] = self.name2label[name]

	end
    return self.subClasses, sub_name2label, all2sub, sub2all
end


function SingleDataSet:get_trimed_dataset(mode)
    local dataset = nil
    if mode == "train" then
        dataset = self.subTrainSet
    elseif mode == "validation" then
        dataset = self.subValSet
    else
        dataset = self.subTestSet
    end

    local posTrimed = {}
    local negTrimed = {}
    for i = 1, #dataset do
        local name = dataset[i]
        local cutName = self:cut_name(name)
        local vidInfo = self.database[cutName]
        local gt_labels = vidInfo["gt_labels"]
        local gt_segments = vidInfo["gt_segments"]
        local duration = vidInfo["duration"]
        for j = 1, gt_labels:size(1) do
            local trimed = {}
            trimed["name"] = 'v_' .. cutName
            trimed["gt_labels"] = self.all2sub[gt_labels[j]]
            trimed["gt_segments"] = gt_segments[j]
            trimed["duration"] = duration
            table.insert(posTrimed, trimed)
            --[[
            local siblings = self:get_siblings(trimed)
            for k = 1, #siblings do
                table.insert(posTrimed, siblings[k])
            end
            ]]
        end
        local neg_segments = self:get_negative_segments(gt_segments)
        if neg_segments:dim() > 0 then
            for j = 1, neg_segments:size(1) do
                local trimed = {}
                trimed["name"] = 'v_' .. cutName
                trimed["gt_labels"] = #self.subClasses + 1
                trimed["gt_segments"] = neg_segments[j]
                trimed["duration"] = duration
                table.insert(negTrimed, trimed)
                --[[
                local siblings = self:get_siblings(trimed)
                for k = 1, #siblings do
                    table.insert(negTrimed, siblings[k])
                end
                ]]
            end
        end
    end
    return posTrimed, negTrimed
end

function SingleDataSet:get_siblings(trimed)
    local segment = trimed['gt_segments']
    local duration = segment[2] - segment[1]
    local insideRatio = 0.66 + torch.rand(2) * (0.95 - 0.66)
    local outsideRatio = 1 - insideRatio
    local leftSeg = {math.max(0, segment[1]-outsideRatio[1]*duration), segment[1]+insideRatio[1]*duration}
    local rightSeg = {segment[2]-insideRatio[2]*duration, math.min(1, segment[2]+outsideRatio[2]*duration)}
    local leftTrimed, rightTrimed = {}, {}
    leftTrimed["name"], rightTrimed["name"] = trimed["name"], trimed["name"]
    leftTrimed["gt_labels"], rightTrimed["gt_labels"] = trimed["gt_labels"], trimed["gt_labels"]
    leftTrimed["duration"], rightTrimed["duration"] = trimed["duration"], trimed["duration"]
    leftTrimed["gt_segments"], rightTrimed["gt_segments"] = torch.Tensor(leftSeg), torch.Tensor(rightSeg)
    return {leftTrimed, rightTrimed}
end


function SingleDataSet:getClassifierBatch(batch_size)
	self.batch_size = batch_size or 18
    local numNeg = #self.negTrimed
    local numPos = #self.posTrimed
    local numPosBatch = self.batch_size
    --local numPosBatch = self.batch_size / 3  -- num of positive samples for each batch
    --local numNegBatch = self.batch_size - numPosBatch         -- num of negative samples for each batch
	local nextBatch = torch.Tensor(self.batch_size, 1, 500)
    local label_targets = torch.Tensor(self.batch_size)
	
	if self.posIdx + numPosBatch - 1 > numPos or self.posIdx == 1 then
		self.posIdx = 1
		self.posRandIdx = torch.randperm(numPos)
	end
    --[[
	if self.negIdx + numNegBatch - 1 > numNeg or self.negIdx == 1 then
		self.negIdx = 1
		self.negRandIdx = torch.randperm(numNeg)
	end
    ]]

    local k = 1
	local limit = self.posIdx + numPosBatch - 1
	for i = self.posIdx, limit  do	
        local trimed = self.posTrimed[i]
        local name = trimed["name"]   -- note: with 'v_' prefix
        local c3ds = self.c3dFile:read(name .. '/c3d_features'):all()
        local c3dseg = self:get_c3d_seg(c3ds,  trimed["gt_segments"], 1)
        --local c3dseg = self:get_c3d_seg_mid(c3ds,  trimed["gt_segments"])
        nextBatch[k] = c3dseg
        label_targets[k] = trimed["gt_labels"]
        k = k + 1
	end
    --[[
	local limit = self.negIdx + numNegBatch - 1
	for i = self.negIdx, limit  do	
        local trimed = self.negTrimed[i]
        local name = trimed["name"]   -- note: with 'v_' prefix
        local c3ds = self.c3dFile:read(name .. '/c3d_features'):all()
        local c3dseg = self:get_c3d_seg(c3ds, trimed["gt_segments"], 1)
        --local c3dseg = self:get_c3d_seg_mid(c3ds, trimed["gt_segments"], 3)
        nextBatch[k] = c3dseg
        label_targets[k] = trimed["gt_labels"]
        k = k + 1
	end
    ]]
    --print("Done..")
	self.posIdx = self.posIdx + numPosBatch
	--self.negIdx = self.negIdx + numNegBatch

	return nextBatch, label_targets

end

function SingleDataSet:getSVMClassifierBatch( batch_size)
	self.batch_size = batch_size or 18
    local numNeg = #self.negTrimed
    local numPos = #self.posTrimed
    local numPosBatch = self.batch_size / 3  -- num of positive samples for each batch
    local numNegBatch = self.batch_size - numPosBatch         -- num of negative samples for each batch
	local nextBatch = torch.Tensor(self.batch_size,  500)
    --local segment_targets = {}
    local label_targets = torch.Tensor(self.batch_size)
	
	if self.posIdx + numPosBatch - 1 > numPos or self.posIdx == 1 then
		self.posIdx = 1
		self.posRandIdx = torch.randperm(numPos)
	end
	if self.negIdx + numNegBatch - 1 > numNeg or self.negIdx == 1 then
		self.negIdx = 1
		self.negRandIdx = torch.randperm(numNeg)
	end

    local k = 1
	local limit = self.posIdx + numPosBatch - 1
	for i = self.posIdx, limit  do	
        local trimed = self.posTrimed[i]
        local name = trimed["name"]   -- note: with 'v_' prefix
        local c3ds = self.c3dFile:read(name .. '/c3d_features'):all()
        local c3dseg = self:get_c3d_seg(c3ds,  trimed["gt_segments"], 1)
        nextBatch[k] = c3dseg:resize(500)
        label_targets[k] = 1
        k = k + 1
	end
	local limit = self.negIdx + numNegBatch - 1
	for i = self.negIdx, limit  do	
        local trimed = self.negTrimed[i]
        local name = trimed["name"]   -- note: with 'v_' prefix
        local c3ds = self.c3dFile:read(name .. '/c3d_features'):all()
        local c3dseg = self:get_c3d_seg(c3ds, trimed["gt_segments"], 1)
        nextBatch[k] = c3dseg:resize(500)
        label_targets[k] = -1
        k = k + 1
	end
    --print("Done..")
	self.posIdx = self.posIdx + numPosBatch
	self.negIdx = self.negIdx + numNegBatch

	return nextBatch, label_targets

end

function SingleDataSet:getMBHClassifierBatch(batch_size)
	self.batch_size = batch_size or 18
    local numNeg = #self.negSet
    local numPos = #self.posSet
    local numPosBatch = self.batch_size / 3  -- num of positive samples for each batch
    local numNegBatch = self.batch_size - numPosBatch         -- num of negative samples for each batch
	local nextBatch = torch.Tensor(self.batch_size,  65536)
    --local segment_targets = {}
    local label_targets = torch.Tensor(self.batch_size)
	
	if self.posIdx + numPosBatch - 1 > numPos or self.posIdx == 1 then
		self.posIdx = 1
		self.posRandIdx = torch.randperm(numPos)
	end
	if self.negIdx + numNegBatch - 1 > numNeg or self.negIdx == 1 then
		self.negIdx = 1
		self.negRandIdx = torch.randperm(numNeg)
	end

    local k = 1
	local limit = self.posIdx + numPosBatch - 1
	for i = self.posIdx, limit  do	
        local name = self.posSet[i]   -- note: with 'v_' prefix
        local path = "features/" .. name
        local mbh = self.mbhFile:read(path):all()
        nextBatch[k] = mbh
        label_targets[k] = 1
        k = k + 1
	end
	local limit = self.negIdx + numNegBatch - 1
	for i = self.negIdx, limit  do	
        local name = self.negSet[i]   -- note: with 'v_' prefix
        local path = "features/" .. name
        local mbh = self.mbhFile:read(path):all()
        nextBatch[k] = mbh
        label_targets[k] = -1
        k = k + 1
	end
    --print("Done..")
	self.posIdx = self.posIdx + numPosBatch
	self.negIdx = self.negIdx + numNegBatch

	return nextBatch, label_targets

end

function SingleDataSet:get_c3d_seg(c3ds,  seg, num)
    -- get three c3d feature vector from the video located in seg
    local numC3d = c3ds:size(1)
    local stFrame, edFrame = math.floor(seg[1] * numC3d), math.floor(seg[2] * numC3d)
    stFrame = (stFrame >= 1 and stFrame) or 1
    stFrame = (stFrame <= numC3d and stFrame) or numC3d
    edFrame = (edFrame >= 1 and edFrame) or 1
    edFrame = (edFrame <= numC3d and edFrame) or numC3d
    local randpos = torch.rand(num) * (edFrame - stFrame)
    local c3d = torch.Tensor(num, 500)
    for i = 1, randpos:size(1) do
        local frame = math.floor(stFrame + randpos[i])
        c3d[i] = c3ds[{{frame}, {}}] 
    end
    return c3d
end
function SingleDataSet:get_c3d_seg_mid(c3ds,  seg)
    -- get three c3d feature vector from the video located in seg
    local mid = (seg[1] + seg[2]) / 2
    local numC3d = c3ds:size(1)
    local frame = math.floor(numC3d * mid)
    if frame <= 1 then frame = 1 end
    if frame >= numC3d then frame = numC3d end
    return c3ds[{{frame}, {}}]
end

function SingleDataSet:get_negative_segments(gt_segments)
    local start = gt_segments[{{}, {1}}]
    local val, idx = torch.sort(start, 1)
    local segs = {}
    local pos = 0
    for i = 1, idx:size(1) do
        local st, ed = gt_segments[idx[i][1]][1], gt_segments[idx[i][1]][2]
        if st - pos >= 0.05 then
            table.insert(segs, {pos, st}) 
        end
        pos = ed
    end
    if 1.0 - pos >= 0.05 then
        table.insert(segs, {pos, 1.0})
    end
    return torch.Tensor(segs)
end

function SingleDataSet:getValSet()
    return self.valSet
end

function SingleDataSet:getTestSet()
    return self.testSet
end

function SingleDataSet:getClassNameByLabel(label)
    return self.label2name[label]
end

function SingleDataSet:getClassLabelByName(clsName)
    return self.name2label[clsName]
end

function SingleDataSet:getSingleSample(vidName, feat_type)
    -- Get c3d(0) or MBH(1) features and ground truth label and segments for a single video
    feat_type = feat_type or 0
    local cutName =  self:cut_name(vidName)
    local vidFeature = nil
    if feat_type == 0 then
        vidFeature = self.c3dFile:read('v_' .. cutName .. '/c3d_features'):all()
    else
        vidFeature = self.mbhFile:read('features/' .. vidName):all()
    end
    local label_target = self.database[cutName]["gt_labels"]
    local segment_target = self.database[cutName]["gt_segments"]
    return vidFeature, label_target, segment_target
end

function SingleDataSet:getSingleTrimedByName(vidName, segments, num_frames)
    local cutName = self:cut_name(vidName)
    local vidFeature = self.c3dFile:read('v_' .. cutName .. '/c3d_features'):all()
    local cutC3d = nil
    if not num_frames then
        cutC3d = self:get_c3d_seg_mid(vidFeature, segments)
        --[[
        num_frames = 0.1 * vidFeature:size(1) * (segments[2] - segments[1])
        num_frames = math.floor(num_frames)
        num_frames = math.max(num_frames, 5)
        ]]
    else
        cutC3d = self:get_c3d_seg(vidFeature, segments, num_frames)
    end
    local label_target = self.database[cutName]["gt_labels"]
    local segment_target = self.database[cutName]["gt_segments"]
    return cutC3d, label_target, segment_target
end

function SingleDataSet:getSingleTrimedSample(num_frames)
    -- Get c3d features and ground truth label for a single trimed video
    if self.posIdx > #self.posTrimed then
        return nil, nil, nil
    end
    if self.posIdx == 1 then
		self.posRandIdx = torch.randperm(#self.posTrimed)
    end

    local trimed = self.posTrimed[self.posRandIdx[self.posIdx]]
    local name = trimed["name"]
    local gt_segments = trimed['gt_segments']
    local c3ds = self.c3dFile:read(name .. '/c3d_features'):all()
    local c3dseg = nil
    if not num_frames then
        c3dseg = self:get_c3d_seg_mid(c3ds, gt_segments, num_frames)
        --[[
        num_frames = 0.1 * c3ds:size(1) * (gt_segments[2] - gt_segments[1])
        num_frames = math.floor(num_frames)
        num_frames = math.max(num_frames, 3)
        ]]
    else
        c3dseg = self:get_c3d_seg(c3ds, gt_segments, num_frames)
    end
    self.posIdx = self.posIdx + 1
    return c3dseg, trimed["gt_labels"], gt_segments
end

function SingleDataSet:getVideoInfo(vidName)
    print(vidName)
    local cutName = self:cut_name(vidName)
    return self.database[cutName]
end

function SingleDataSet:parse_json()
    local f = io.open(self.jsPath, 'r')
    for line in f:lines() do
        json_text = line
	    break
    end
    local json = cjson.decode(json_text)
    return json["database"], json["version"], json["taxonomy"]
end

function SingleDataSet:create_dataset()
--  Get DataSet
    trainSet, valSet, testSet = {}, {}, {}
    for i = 1, #self.cutNameList do
        name = self.cutNameList[i]
        if not self.database[name] then
	    print("Video " .. name.. " not found in json")
	elseif self.database[name]['subset'] == "training" then
	    table.insert(trainSet, self.videoList[i])
	elseif self.database[name]['subset'] == "validation" then
	    table.insert(valSet, self.videoList[i])
	elseif self.database[name]['subset'] == 'testing' then
	    table.insert(testSet, self.videoList[i])
	end

        -- Get GT labels and segments
	local annotations = self.database[name]["annotations"]
	local duration = self.database[name]["duration"]
	local gt_segments = torch.Tensor(#annotations, 2)
	local gt_labels = torch.Tensor(#annotations)
    if #annotations > 0 then
        for j = 1, #annotations do
            local segment = annotations[j]["segment"]
            local st, ed = segment[1] / duration, segment[2] / duration
            gt_segments[j] = torch.Tensor({st, ed})
            local labelName = annotations[j]["label"]
            gt_labels[j] = self.name2label[labelName]
        end
    else
        gt_segments = torch.zeros(1, 2)
        gt_labels = torch.Tensor(1):fill(201)
    end
	self.database[name]["gt_segments"] = gt_segments
	self.database[name]["gt_labels"] = gt_labels
    end
    return trainSet, valSet, testSet
end


function SingleDataSet:cut_name(name)
	name = string.sub(name, 3, -1)
	return utils.split(name, ".")[1]
end


-- get next batch containing several video tensors
function SingleDataSet:getNextBatch(batch_size)
	self.batch_size = batch_size or 18
    local numPos = #self.subTrainSet 
    local numPosBatch = self.batch_size    -- num of positive samples for each batch
	local nextBatch = {}
    local segment_targets = {}
    local label_targets = {}
	
	if self.posIdx + numPosBatch - 1 > numPos or self.posIdx == 1 then
		self.posIdx = 1
		self.posRandIdx = torch.randperm(numPos)
	end
    --[[
	if self.negIdx + numNegBatch - 1 > numNeg or self.negIdx == 1 then
		self.negIdx = 1
		self.negRandIdx = torch.randperm(numNeg)
	end
    ]]

	local limit = self.posIdx + numPosBatch - 1
	for i = self.posIdx, limit  do	
        local vidName = self.subTrainSet[ self.posRandIdx[i]]
        local cutName = self:cut_name(vidName)
        table.insert(nextBatch, self.c3dFile:read('v_' .. cutName .. '/c3d_features'):all())
        table.insert(segment_targets, self.database[cutName]["gt_segments"])
        local gt_labels = self.database[cutName]["gt_labels"]
        local fix_gt_labels = {}
        for j = 1, gt_labels:size(1) do
            fix_gt_labels[j] = self.all2sub[gt_labels[j]]
        end
        table.insert(label_targets, torch.Tensor(fix_gt_labels))
	end
    --[=[
	local limit = self.negIdx + numNegBatch - 1
	for i = self.negIdx, limit  do	
        local vidName = self.negSet[ self.negRandIdx[i]]
        local cutName = self:cut_name(vidName)
        table.insert(nextBatch, self.c3dFile:read('v_' .. cutName .. '/c3d_features'):all())
        table.insert(segment_targets, torch.zeros(1, 2))
        table.insert(label_targets, torch.Tensor(1):fill(#self.subClasses+1))
	end
    ]=]
    --print("Done..")
	self.posIdx = self.posIdx + numPosBatch
	--self.negIdx = self.negIdx + numNegBatch

	return nextBatch, label_targets, segment_targets
end

function SingleDataSet:get_negative_set(posSet)
    local posSet = utils.Set(posSet)
    local negSet = {}
    for i = 1, #self.trainSet do
        if not posSet[self.trainSet[i]] then
            table.insert(negSet, self.trainSet[i])
        end
        --[[
        if #negSet == 2 * #posSet then
            break
        end
        ]]
    end
    return negSet
end


function SingleDataSet:check_file()
	for i=1, #self.videoList do
		local succ = io.open('/media/lwm/wjz/data/Videos/activitynet/data/' .. self.videoList[i], 'r')
		if nil == succ then
			print(i .. ': ' .. self.videoList[i])
		else
			io.close(succ)
		end
	end
end

function SingleDataSet:get_classes()
	local name2label = {}
	local label2name = {}
	local cnt = 1
	for id, taxon in pairs(self.id2Tax) do
		if #taxon["child"] == 0 then
			name2label[taxon["nodeName"]] = cnt
			label2name[cnt] = taxon["nodeName"]
			cnt = cnt + 1
		end
	end
	return name2label, label2name
end

function SingleDataSet:create_taxonomy_tree()
	self.id2Tax = {}  -- nodeId -> taxonomy
	self.subTree = {}  --> subtree rooted from nodeId
	for i = 1, #self.taxonomy do
	--local id = self.taxonomy[i]["nodeId"]
		local nodeId = self.taxonomy[i]["nodeId"]
		self.id2Tax[nodeId] = self.taxonomy[i]
		local parentId = self.taxonomy[i]["parentId"]
		if not self.subTree[parentId] then
			self.subTree[parentId] = {}
		end
	    table.insert(self.subTree[parentId], nodeId)
	end
	local root = self:recursive_contruct_tree(0)
	return root, self.id2Tax
end

function SingleDataSet:recursive_contruct_tree(rootid)
	local child = {}
	-- note: id2Tax will be changed along with annotation( child added)
	local annotation = self.id2Tax[rootid]
	local subroot = self.subTree[rootid]
	if not subroot then
		self.subTree[rootid] = {}
		annotation["child"] = child
		return annotation
	end
	for _, cid in pairs(subroot) do
		table.insert(child, self:recursive_contruct_tree(cid)) 
	end
	annotation["child"] = child
	return annotation
end

function SingleDataSet:retrieval_all_leaves(subroot)
    if #subroot.child == 0 then
        return {subroot.nodeName}
    end
    local names = {}
    for i = 1, #subroot.child do
        local cNames = self:retrieval_all_leaves(subroot.child[i])
        table.foreach(cNames, function(i, v) table.insert(names, v) end)
    end
    return names
end

function SingleDataSet:create_video_list()
	local f = io.open(self.idxPath, 'r') 
	local videoList, cutNameList = {}, {}
	assert(f)
	local cnt = 0
	for line in f:lines() do
		local re = utils.split(line, ",")
		table.insert(videoList, re[3])
		table.insert(cutNameList, self:cut_name(re[3]))
	end
	f:close()
    
        
	return videoList, cutNameList
end

function SingleDataSet:destroy()
    self.c3dFile:close()
    --self.mbhFile:close()
end

