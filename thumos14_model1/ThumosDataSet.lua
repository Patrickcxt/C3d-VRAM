-----------------------------------------------------------------------
--[[ "Sports" subset of ThumosDataSet ]]--
--Integrating some operations to deal with video data.
-----------------------------------------------------------------------
--
--require 'ffmpeg'
require 'image'
require 'utils'
require 'lfs'
require 'hdf5'
local cjson = require 'cjson'
--local torchvid = require 'torchvid'

local ThumosDataSet = torch.class("ThumosDataSet")

function ThumosDataSet:__init(rootPath, idxPath, jsPath) -- /path/to/your/Videos/
	self.rootPath = rootPath or '/home/amax/cxt/data/THUMOS2014/'
    self.trainIdtFile = hdf5.open(self.rootPath .. "training/ucf_101/idts_bof.h5", 'r')
    self.valIdtFile = hdf5.open(self.rootPath .. "validation/TH14_validation_features/idts_bof.h5", 'r')
    self.valAnnotationFile = hdf5.open(self.rootPath .. 'validation/video_infos.h5', 'r')
    self.testAnnotationFile = hdf5.open(self.rootPath .. 'test/video_infos.h5', 'r')
    self.valFeaturesFile = hdf5.open(self.rootPath .. 'validation/sports_features3.h5', 'r')
    self.testFeaturesFile = hdf5.open(self.rootPath .. 'test/sports_features3.h5', 'r')

    self.label2name, self.name2label = self:get_classes()
    self.trainVideos, self.trainGT = self:create_trainset()
	self.valVideos, self.valGT, self.valSegments, self.valSegmentLabel= self:create_valset()
    self.testVideos, self.testGT, self.testSegments, self.testSegmentLabel = self:create_testset()
    print("Num of validation videos: ", #self.valVideos)
    print("Num of test videos: ", #self.testVideos)
    self._iter = 1

end

function ThumosDataSet:get_classes()

	local name2label = {}
    local label2name = {
     "BaseballPitch", "BasketballDunk", "Billiards", "CleanAndJerk", "CliffDiving",
     "CricketBowling", "CricketShot", "Diving","FrisbeeCatch", "GolfSwing",
     "HammerThrow", "HighJump", "JavelinThrow", "LongJump", "PoleVault",
     "Shotput", "SoccerPenalty", "TennisSwing", "ThrowDiscus", "VolleyballSpiking"
    }
    for i = 1, #label2name do
        name2label[label2name[i]] = i
    end
	return label2name, name2label
end

function ThumosDataSet:create_trainset()
    -- get ground truth
    local labels = {}
    local videoList = {}
    local path = self.rootPath .. "training/ucf_101/"
    --local myFile = hdf5.open(path .. 'idts_bof.h5', 'w')
    for fn in lfs.dir(path) do
        local class_name = utils.split(fn, '_')[2]
        if self.name2label[class_name] then
            fn = string.sub(fn, 1, -5)
            table.insert(videoList, fn)
            table.insert(labels, self.name2label[class_name])
            --print("Processing " .. fn)
            --local vector = self:construct_single_sample(fn)
            --myFile:write(fn, vector)
        end
    end
    --myFile:close()
    return videoList, labels
end

function ThumosDataSet:create_valset()
    -- get ground truth
    local database = {}
    local videoList = {}
    for i = 1, #self.label2name do
        --print("Processing " .. self.label2name[i])
        local filename = self.rootPath .. "validation/annotation/" .. self.label2name[i] .. '_val.txt'
        local f = io.open(filename, 'r')
        for line in f:lines() do
            local infos = utils.split(line , ' ')
            local vidName, st, ed = unpack(infos) 
            if not database[vidName] then
                database[vidName] = {}
                table.insert(videoList, vidName)
            end
            local instance = {}
            --instance["label"], instance["st"], instance["ed"] = self.label2name[i], st, ed
            instance["label"], instance["st"], instance["ed"] = i, st, ed
            table.insert(database[vidName], instance)
        end
    end

    local path = self.rootPath .. "validation/TH14_validation_features/"
    --local myFile = hdf5.open(path .. 'idts_bof.h5', 'w')
    local valSet = {}
    local valLabel = {}
    for video_name, instances in pairs(database) do
        --local fn = path .. video_name .. '.txt'
        --print("Processing idts of " .. fn)
        for i = 1, #instances do
            local label, st, ed = instances[i]["label"], instances[i]["st"],  instances[i]["ed"]
            local name = video_name .. '_' .. tostring(i)
            --[[
            local annotation = self.valAnnotationFile:read(video_name):all()
            local duration, frame_num = annotation[1], annotation[3]
            print(label, st, ed, duration, frame_num)
            st, ed = frame_num*(st/duration), frame_num*(ed / duration)
            print(st, ed)
            local vector = self:construct_single_sample_bound(fn, st, ed)
            myFile:write(name, vector)
            ]]
            table.insert(valSet, name)
            table.insert(valLabel, label)
        end
    end
    --myFile:close()
    return videoList, database, valSet, valLabel
end

function ThumosDataSet:create_testset()
    -- get ground truth
    local database = {}
    local videoList = {}
    for i = 1, #self.label2name do
        --print("Processing " .. self.label2name[i])
        local filename = self.rootPath .. "test/annotation/" .. self.label2name[i] .. '_test.txt'
        local f = io.open(filename, 'r')
        for line in f:lines() do
            local infos = utils.split(line , ' ')
            local vidName, st, ed = unpack(infos) 
            if not database[vidName] then
                database[vidName] = {}
                table.insert(videoList, vidName)
            end
            local instance = {}
            --instance["label"], instance["st"], instance["ed"] = self.label2name[i], st, ed
            instance["label"], instance["st"], instance["ed"] = i, st, ed
            table.insert(database[vidName], instance)
        end
    end

    local path = self.rootPath .. "test/TH14_test_features/"
    --local myFile = hdf5.open(path .. 'idts_bof.h5', 'w')
    local testSet = {}
    local testLabel = {}
    local k = 1
    for video_name, instances in pairs(database) do
        --local fn = path .. video_name .. '.txt'
        --print(tostring(k) .. " Processing idts of " .. fn)
        k = k + 1
        for i = 1, #instances do
            local label, st, ed = instances[i]["label"], instances[i]["st"],  instances[i]["ed"]
            local name = video_name .. '_' .. tostring(i)
            --[[
            local annotation = self.testAnnotationFile:read(video_name):all()
            local duration, frame_num = annotation[1], annotation[3]
            print(label, st, ed, duration, frame_num)
            st, ed = frame_num*(st/duration), frame_num*(ed / duration)
            print(st, ed)
            local vector = self:construct_single_sample_bound(fn, st, ed)
            myFile:write(name, vector)
            ]]
            table.insert(testSet, name)
            table.insert(testLabel, label)
        end
    end
    --myFile:close()
    return videoList, database, testSet, testLabel
end

-- get next batch containing several video names
function ThumosDataSet:getNextBatch(batch_size)
	self.batch_size = batch_size or 18
	local nextBatch ={}
    local label_targets = {}
    local segment_targets = {}
	
	if self._iter + self.batch_size - 1 > #self.valVideos or self._iter == 1 then
		self._iter = 1
		self._perm = torch.randperm(#self.valVideos)
	end

    local k = 1
	local limit = self._iter + self.batch_size - 1
	for i = self._iter, limit  do	
        local video_name = self.valVideos[self._perm[i]]
        local annotation = self.valAnnotationFile:read(video_name):all()
        local duration = annotation[1]

        nextBatch[k] = self.valFeaturesFile:read(video_name):all()

        local instances = self.valGT[video_name]
        local labels = torch.Tensor(#instances)
        local segments = torch.Tensor(#instances, 2)
        for j = 1, #instances do
            local l, s, e = instances[j]['label'], instances[j]['st'], instances[j]['ed']
            labels[j] = l
            segments[j][1], segments[j][2] = s/duration, e/duration
        end
        label_targets[k] = labels
        segment_targets[k] = segments
        k = k + 1
	end
	self._iter = self._iter + self.batch_size

	return nextBatch, label_targets, segment_targets
end

function ThumosDataSet:getValSample()
    if self._iter == 1 then
        self._perm = torch.randperm(#self.valVideos)
    end
    if self._iter > #self.valVideos then
        --self._iter = 1
        return nil
    end
    local video_name = self.valVideos[self._perm[self._iter]]
    print(video_name)
    local annotation = self.valAnnotationFile:read(video_name):all()
    local feature = self.valFeaturesFile:read(video_name):all()

    local duration = annotation[1]
    local instances = self.valGT[video_name]
    local labels = torch.Tensor(#instances)
    local segments = torch.Tensor(#instances, 2)
    for j = 1, #instances do
        local l, s, e = instances[j]['label'], instances[j]['st'], instances[j]['ed']
        labels[j] = l
        segments[j][1], segments[j][2] = s/duration, e/duration
    end
    self._iter = self._iter + 1

    return video_name, annotation, feature, labels, segments
end

function ThumosDataSet:getTestSample()
    if self._iter == 1 then
        self._perm = torch.randperm(#self.testVideos)
    end
    if self._iter > #self.testVideos then
        return nil
    end
    local video_name = self.testVideos[self._perm[self._iter]]
    print(video_name)
    local annotation = self.testAnnotationFile:read(video_name):all()
    local feature = self.testFeaturesFile:read(video_name):all()

    local duration = annotation[1]
    local instances = self.testGT[video_name]
    local labels = torch.Tensor(#instances)
    local segments = torch.Tensor(#instances, 2)
    for j = 1, #instances do
        local l, s, e = instances[j]['label'], instances[j]['st'], instances[j]['ed']
        labels[j] = l
        segments[j][1], segments[j][2] = s/duration, e/duration
    end
    self._iter = self._iter + 1

    return video_name, annotation, feature, labels, segments
end

function ThumosDataSet:get_idt_seg_mid(idts,  seg)
    -- get three c3d feature vector from the video located in seg
    local mid = (seg[1] + seg[2]) / 2
    local num_segment = idts:size(1)
    local frame = math.floor(num_segment * mid)
    if frame <= 1 then frame = 1 end
    if frame >= num_segment then frame = num_segment end
    return idts[frame]
end

function ThumosDataSet:get_idt_seg(idts,  seg)
    local num_segment = idts:size(1)
    local f1 = math.floor(num_segment*seg[1])
    local f2 = math.floor(num_segment*seg[2])
    f1, f2 = math.max(1, f1), math.max(1, f2)
    f1, f2 = math.min(num_segment, f1), math.min(num_segment, f2)
    --print(f1, f2)
    local features = torch.zeros(16000)
    for i = f1, f2, 2 do
        features = features + idts[i]
    end
    --local num_idts = features[{{1, 4000}}]:sum()
    --features = features:div(num_idts)
    return features
end

--[[
function ThumosDataSet:plot_idt_density(video_name,  seg, num_frames)
    local f = io.open('/home/amax/cxt/data/THUMOS2014/test/TH14_test_features/' .. video_name .. '.txt', 'r')
    local x = torch.IntTensor(num_frames):fill(0)
    for line in f:lines() do
        local infos = utils.split(line, '\t')
        local frame = tonumber(infos[1])
        x[frame] = x[frame] + 1
    end
    gnuplot.plot(x)
    return nil
end
]]

function ThumosDataSet:plot_idt_density(idts)
    local x = torch.IntTensor(idts:size(1)):fill(0)
    local acc_x = torch.IntTensor(idts:size(1)):fill(0)
    for i = 1, idts:size(1) do
        x[i] = idts[i][{{1, 4000}}]:sum()
        if i == 1 then acc_x[i] = x[i]
        else acc_x[i] = acc_x[i-1] + x[i] end
    end
    --gnuplot.plot(x)
    return x, acc_x
end

--[[
function ThumosDataSet:get_idt_seg_raw(idts,  seg)
    local num_segment = idts:size(1)
    local f1 = math.floor(num_segment*seg[1])
    local f2 = math.floor(num_segment*seg[2])
    f1, f2 = math.max(1, f1), math.max(1, f2)
    f1, f2 = math.min(num_segment, f1), math.min(num_segment, f2)
    --print(f1, f2)
    local features = torch.zeros(16000)
    local k = 0
    for i = f1, f2, 2 do
        features = features + idts[i]
        k = k + 1
    end
    return features, k
end
]]

function ThumosDataSet:getClassifierBatch(batch_size)
	self.batch_size = batch_size or 18
	local nextBatch = torch.Tensor(self.batch_size, 16000) -- batch_size x num of idt x dimention of idt feature
    local label_targets = torch.Tensor(self.batch_size)
	
	if self._iter + self.batch_size - 1 > #self.trainVideos or self._iter == 1 then
		self._iter = 1
		self._perm = torch.randperm(#self.trainVideos)
	end

    local k = 1
	local limit = self._iter + self.batch_size - 1
	for i = self._iter, limit  do	
        local video_name = self.trainVideos[self._perm[i]]
        --nextBatch[k] = self:construct_single_sample(video_name)
        nextBatch[k] = self.trainIdtFile:read(video_name):all()
        label_targets[k] = self.trainGT[self._perm[i]]
        k = k + 1
	end
	self._iter = self._iter + self.batch_size
	return nextBatch, label_targets

end


function ThumosDataSet:getSingleValSample()
    if self._iter == 1 then
        self._perm = torch.randperm(#self.valVideos)
    end
    if self._iter > #self.valVideos then
        return nil
    end
    local video_name = self.valVideos[self._perm[self._iter]]
    --local iDT = self:construct_single_sample(video_name)
    iDT = self.valIdtFile:read(video_name):all()
    local label = self.valGT[self._perm[self._iter]]
    self._iter = self._iter + 1
    return iDT, label
end

function ThumosDataSet:getSingleTestSample()
    if self._iter == 1 then
        self._perm = torch.randperm(#self.testVideos)
    end
    if self._iter > #self.testVideos then
        return nil
    end
    local video_name = self.testVideos[self._perm[self._iter]]
    --local iDT = self:construct_single_sample(video_name)
    iDT = self.valIdtFile:read(video_name):all()
    local label = self.testGT[self._perm[self._iter]]
    self._iter = self._iter + 1
    print(iDT)
    print(video_name)
    print(label)
    io.read()
    return iDT, label
end

function ThumosDataSet:construct_single_sample(video_name)
    -- Randomly sample 256 iDT in a video to construct a 1024-dimentional feature vector
    local path = self.rootPath .. "training/ucf_101/" .. video_name .. ".txt"
    local f = io.open(path, 'r')
    local iDTs = {}
    for line in f:lines() do
        local infos = utils.split(line, '\t') 
        local feature_vector = torch.Tensor({infos[4], infos[5], infos[6], infos[7]})
        feature_vector:add(1)
        table.insert(iDTs, feature_vector)
    end
    local vector = self:BoFPooling(iDTs)
    return vector
end

function ThumosDataSet:construct_single_sample_bound(video_path, st, ed)
    local f = io.open(video_path, 'r')
    local iDTs = {}
    for line in f:lines() do
        local infos = utils.split(line, '\t') 
        local frame = tonumber(infos[1])
        if frame >= st and frame <= ed then
            local feature_vector = torch.Tensor({infos[4], infos[5], infos[6], infos[7]})
            feature_vector:add(1)
            table.insert(iDTs, feature_vector)
        end
    end
    local vector = self:BoFPooling(iDTs)
    return vector
end

function ThumosDataSet:BoFPooling(iDTs)
    local Traj, Hog, Hof, Mbh = torch.zeros(4000), torch.zeros(4000), torch.zeros(4000), torch.zeros(4000)
    for i = 1, #iDTs do
        local traj, hog, hof, mbh = iDTs[i][1],iDTs[i][2], iDTs[i][3], iDTs[i][4]
        Traj[traj] = Traj[traj] + 1
        Hog[hog] = Hog[hog] + 1
        Hof[hof] = Hof[hof] + 1

        Mbh[mbh] = Mbh[mbh] + 1
    end
    local feat = Traj:cat(Hog):cat(Hof):cat(Mbh)
    feat:div(#iDTs)
    return feat
end


---------------------------- Not used now ------------------------------------------------------


function ThumosDataSet:setClass(clsName, mode)
    self.clsName = clsName
    self.subClasses, self.subTrainSet, self.subValSet, 
        self.subTestSet, self.negSet = self:getSubSet()
    print(self.subClasses)
    io.read()
	self.sub_label2name, self.sub_name2label, self.all2sub, self.sub2all = self:get_sub_classes()
    self.posTrimed, self.negTrimed = self:get_trimed_dataset(mode)
    print("num of pos trimed: ", #self.posTrimed)
	self.posIdx = 1
    self.negIdx = 1
    return self.subTrainSet, self.subValSet, self.subTestSet, self.negSet
end

function ThumosDataSet:seperateOneVsAll(clsLabel)
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

function ThumosDataSet:seperateOneVsAllGlobal(clsLabel, mode)
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

function ThumosDataSet:getMapping()
    return self.sub_label2name, self.sub_name2label,  self.all2sub, self.sub2all
end

function ThumosDataSet:getSubSet()
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


function ThumosDataSet:get_sub_classes()
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


function ThumosDataSet:get_trimed_dataset(mode)
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

function ThumosDataSet:get_siblings(trimed)
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



function ThumosDataSet:getSVMClassifierBatch( batch_size)
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

function ThumosDataSet:getMBHClassifierBatch(batch_size)
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

function ThumosDataSet:get_c3d_seg(c3ds,  seg, num)
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

function ThumosDataSet:get_negative_segments(gt_segments)
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

function ThumosDataSet:getValSet()
    return self.valSet
end

function ThumosDataSet:getTestSet()
    return self.testSet
end

function ThumosDataSet:getClassNameByLabel(label)
    return self.label2name[label]
end

function ThumosDataSet:getClassLabelByName(clsName)
    return self.name2label[clsName]
end

function ThumosDataSet:getSingleSample(vidName, feat_type)
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


function ThumosDataSet:getSingleTrimedSample(num_frames)
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

function ThumosDataSet:getVideoInfo(vidName)
    local cutName = self:cut_name(vidName)
    return self.database[cutName]
end

function ThumosDataSet:parse_json()
    local f = io.open(self.jsPath, 'r')
    for line in f:lines() do
        json_text = line
	    break
    end
    local json = cjson.decode(json_text)
    return json["database"], json["version"], json["taxonomy"]
end



function ThumosDataSet:cut_name(name)
	name = string.sub(name, 3, -1)
	return utils.split(name, ".")[1]
end



function ThumosDataSet:get_negative_set(posSet)
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


function ThumosDataSet:check_file()
	for i=1, #self.videoList do
		local succ = io.open('/media/lwm/wjz/data/Videos/activitynet/data/' .. self.videoList[i], 'r')
		if nil == succ then
			print(i .. ': ' .. self.videoList[i])
		else
			io.close(succ)
		end
	end
end


function ThumosDataSet:create_taxonomy_tree()
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

function ThumosDataSet:recursive_contruct_tree(rootid)
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

function ThumosDataSet:retrieval_all_leaves(subroot)
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

function ThumosDataSet:create_video_list()
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

function ThumosDataSet:destroy()
    self.c3dFile:close()
    --self.mbhFile:close()
end


