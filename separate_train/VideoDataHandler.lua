-----------------------------------------------------------------------
--[[ VideoDataHandler ]]--
--Integrating some operations to deal with video data.
-----------------------------------------------------------------------
require 'ffmpeg'
require 'image'
require 'utils'
require 'lfs'
require 'hdf5'
local cjson = require 'cjson'
--local torchvid = require 'torchvid'

local VideoDataHandler = torch.class("VideoDataHandler")

function VideoDataHandler:__init(rootPath, idxPath, jsPath) -- /path/to/your/Videos/
	self.rootPath = rootPath or '/media/caffe/wjz1/data/Videos/activitynet/'
    self.idxPath = idxPath or '/home/caffe/cxt/data/ActivityNet/scheme/index.txt'
    self.jsPath = jsPath or "/home/caffe/cxt/data/ActivityNet/scheme/activity_net.v1-3.min.json"
    self.cachePath = "/home/caffe/cxt/VRAM/cache/"
    self.c3dPath = "/home/caffe/cxt/data/ActivityNet/features/sub_activitynet_v1-3.c3d.hdf5"
    --self.c3dPath = "/home/caffe/cxt/data/ActivityNet/features/ds_activitynet_v1-3.c3d.hdf5" -- downsample

	self.videoList, self.cutNameList = self:create_video_list()
	self.database, self.version, self.taxonomy = self:parse_json()
	self.root, self.id2Tax = self:create_taxonomy_tree()
    self.name2label, self.label2name = self:get_classes()
	self.trainSet, self.valSet, self.testSet = self:create_dataset()
	self.posIdx = 1
    self.negIdx = 1
    self.c3dFile = hdf5.open(self.c3dPath, 'r')
end

function VideoDataHandler:getSubSet(clsName)
    if clsName == nil then
        return #self.label2name, self.trainSet, self.valSet
    end
    local subClasses = {}
    local trainSet = {}
    local valSet = {}
    local testSet = {}
	for id, taxon in pairs(self.id2Tax) do
        if taxon.nodeName == clsName then
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
    self.subTrainSet = trainSet
    self.subValSet = valSet
    self.negSet = self:get_negative_set(self.subTrainSet)
    return self.subClasses, self.subTrainSet, self.subValSet
end

function VideoDataHandler:getValSet()
    return self.valSet
end

function VideoDataHandler:getTestSet()
    return self.testSet
end

function VideoDataHandler:getClassNameByLabel(label)
    return self.label2name[label]
end

function VideoDataHandler:getClassLabelByName(clsName)
    return self.name2label[clsName]
end

function VideoDataHandler:getSingleSample(vidName)
    local cutName =  self:cut_name(vidName)
    local vidFeature = self.c3dFile:read('v_' .. cutName .. '/c3d_features'):all()
    local label_target = self.database[cutName]["gt_labels"]
    local segment_target = self.database[cutName]["gt_segments"]
    return vidFeature, label_target, segment_target
end

function VideoDataHandler:getVideoInfo(vidName)
    local cutName = self:cut_name(vidName)
    return self.database[cutName]
end

function VideoDataHandler:parse_json()
    local f = io.open(self.jsPath, 'r')
    for line in f:lines() do
        json_text = line
	    break
    end
    local json = cjson.decode(json_text)
    return json["database"], json["version"], json["taxonomy"]
end

function VideoDataHandler:create_dataset()
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


function VideoDataHandler:cut_name(name)
	name = string.sub(name, 3, -1)
	return utils.split(name, ".")[1]
end


-- get next batch containing several video tensors
function VideoDataHandler:getNextBatch(batch_size)
	self.batch_size = batch_size or 18
    local numNeg = #self.negSet
    local numPos = #self.subTrainSet 
    local numPosBatch = self.batch_size / 3  -- num of positive samples for each batch
    local numNegBatch = 2 * numPos         -- num of negative samples for each batch
	local nextBatch = {}
    local segment_targets = {}
    local label_targets = {}
	
	if self.posIdx + numPosBatch - 1 > numPos or self.posIdx == 1 then
		self.posIdx = 1
		self.posRandIdx = torch.randperm(numPos)
	end
	if self.negIdx + numNegBatch - 1 > numNeg or self.negIdx == 1 then
		self.negIdx = 1
		self.negRandIdx = torch.randperm(numNeg)
	end

	local limit = self.posIdx + numPosBatch - 1
	for i = self.posIdx, limit  do	
        local vidName = self.subTrainSet[ self.posRandIdx[i]]
        local cutName = self:cut_name(vidName)
        table.insert(nextBatch, self.c3dFile:read('v_' .. cutName .. '/c3d_features'):all())
        table.insert(segment_targets, self.database[cutName]["gt_segments"])
        table.insert(label_targets, self.database[cutName]["gt_labels"])
	end
	local limit = self.negIdx + numNegBatch - 1
	for i = self.negIdx, limit  do	
        local vidName = self.negSet[ self.negRandIdx[i]]
        local cutName = self:cut_name(vidName)
        table.insert(nextBatch, self.c3dFile:read('v_' .. cutName .. '/c3d_features'):all())
        table.insert(segment_targets, torch.zeros(1, 2))
        table.insert(label_targets, torch.Tensor(1):fill(#self.subClasses+1))
	end
    --print("Done..")
	self.posIdx = self.posIdx + numPosBatch
	self.negIdx = self.negIdx + numNegBatch

	return nextBatch, label_targets, segment_targets
end

function VideoDataHandler:get_negative_set(posSet)
    local posSet = utils.Set(posSet)
    local negSet = {}
    for i = 1, #self.trainSet do
        if not posSet[self.trainSet[i]] then
            table.insert(negSet, self.trainSet[i])
        end
        if #negSet == 2 * #posSet then
            break
        end
    end
    return negSet
end


function VideoDataHandler:check_file()
	for i=1, #self.videoList do
		local succ = io.open('/media/lwm/wjz/data/Videos/activitynet/data/' .. self.videoList[i], 'r')
		if nil == succ then
			print(i .. ': ' .. self.videoList[i])
		else
			io.close(succ)
		end
	end
end

function VideoDataHandler:get_classes()
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

function VideoDataHandler:create_taxonomy_tree()
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

function VideoDataHandler:recursive_contruct_tree(rootid)
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

function VideoDataHandler:retrieval_all_leaves(subroot)
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

function VideoDataHandler:create_video_list()
	local f = io.open(self.idxPath, 'r') 
	videoList, cutNameList = {}, {}
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

function VideoDataHandler:destroy()
    self.c3dFile:close()
end

-- Present video as torch.Tensor.
--[[
function VideoDataHandler:readVideo(vid_name)
	if not vid_name then
		print('Argument #1 is needed to specify video.')
		assert(vid_name)
	else
		vid = ffmpeg.Video(self.rootPath .. 'data/' .. vid_name)
		vid_tensor = vid:totensor{}
		return vid_tensor
	end
end
function VideoDataHandler:readVideo(vid_name)
    local vidPath = self.rootPath .. 'data/' .. vid_name
    local vidCachePath = self.cachePath .. 'videos/' .. self:cut_name(vid_name)
    local cmd = "python ReadVideo.py " .. vidPath
    os.execute(cmd)
	if not vid_name then
		print('Argument #1 is needed to specify video.')
		assert(vid_name)
	end
    local imgid = 0
    local frameList = {}
    while true do
        local imgName = string.format(vidCachePath .. "/%06d.jpg", imgid) 
        if not utils.file_exists(imgName) then break end
        local im = image.loadJPG(imgName)
        im:resize(3, 1, 224, 224)
        table.insert(frameList, im)
        imgid = imgid + 1
    end
    local vidTensor = torch.Tensor(3, imgid, 224, 224)
    for i = 1, imgid do
        vidTensor[{{}, {i}, {}, {}}] = frameList[i]
    end
    print("video " .. vid_name ..  " loaded !")
    --vidTensor:resize(imgid, c, h, w)
    return vidTensor
end


-- Resize the depth-width ratio of each frame.
function VideoDataHandler:resizeRatio(vid)
	if not vid then
		print('Argument #1 is nil.')
		assert(vid)
	else
		local frames_num = vid:size(1)
		vid_resize = torch.Tensor( 3, frames_num, 224, 224):zero()
		for idx = 1, frames_num do
			frame = vid[idx]
			frame_resize = image.scale( frame, 224, 224 ):resize( 3, 1, 224, 224) 
			vid_resize[{{},{idx},{},{}}]:copy(frame_resize)
		end
		return vid_resize
	end
end
]]
