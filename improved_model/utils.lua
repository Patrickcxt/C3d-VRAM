-------------------j-----------------------------
-- utility functions for the evaluation part
------------------------------------------------
--local torchvid = require 'torchvid'

utils = {}
function utils.interval_overlap(gts, dets)
  local num_gt = gts:size(1)
  local num_det = dets:size(1)
  local ov = torch.Tensor(num_gt, num_det)
  for i=1,num_gt do
    for j=1,num_det do
      ov[i][j] = utils.interval_overlap_single(gts[i], dets[j])
    end
  end
  return ov
end

function utils.interval_overlap_single(gt, dt)
  local i1 = gt
  local i2 = dt
  -- union
  local bu = {math.min(i1[1], i2[1]), math.max(i1[2], i2[2])}
  local ua = bu[2] - bu[1]
  -- overlap
  local ov = 0
  local bi = {math.max(i1[1], i2[1]), math.min(i1[2], i2[2])}
  local iw = bi[2] - bi[1]
  if iw > 0 then
    ov = iw / ua
  end
  return ov
end

function utils.round(num, idp)
  local mult = 10^(idp or 0)
  return math.floor(num * mult + 0.5) / mult
end

function utils.get_gts(input, target)
    local batch_size = #target
    local reg_target = torch.Tensor(batch_size, 2)
    for i = 1, batch_size do
        local det = torch.Tensor(2)
        det:copy(input[i]):resize(1, 2)
        local gts = target[i]
        local overlaps = utils.interval_overlap(gts, det)
        local max_ov, max_ovid  = torch.max(overlaps, 1)
        if (max_ov[1][1] > 0) then
            reg_target[i] = gts[max_ovid[1][1]]
        else
            local dists = utils.segment_distance(gts, det)    
            local min_dis, min_disid = torch.min(dists, 1)
            reg_target[i] = gts[min_disid[1][1]]
        end
    end
    return reg_target
end

function utils.segment_distance(gts, dets)
    local num_gt = gts:size(1)
    local num_det = dets:size(1)
    local dists = torch.Tensor(num_gt, num_det)
    for i=1,num_gt do
        for j=1,num_det do
            dists[i][j] = utils.segment_distance_single(gts[i], dets[j])
        end
    end
    return dists
end

function utils.segment_distance_single(gt, det)
    local s1, e1 = gt[1], gt[2]
    local s2, e2 = det[1], det[2]
    if (s1 > s2) then
        return math.max(s1 - e2, 0)
    else
        return math.max(s2 - e1, 0)
    end
end

function utils.split(s, splitor)
    local t = {}
    local regexp = "([^'" .. splitor .. "']+)"
    for w in string.gmatch(s, regexp) do 
        table.insert(t, w)
    end
    return t
end

function utils.file_exists(name)
    local f = io.open(name, 'r')
    if f ~= nil then
        io.close(f)
        return true
    end
    return false;
end

function utils.nms(segs, scores, thresh)
    local pick = torch.LongTensor()
    if segs:numel() == 0 then
        return pick
    end
    local vals, I = scores:sort(1)
    pick:resize(scores:size()):zero()
    local counter = 1
    while I:numel() > 0 do
        local last = I:size(1)
        local i = I[last]
        pick[counter] = i
        counter = counter + 1
        if last == 1 then
            break
        end
        I = I[{{1, last-1}}]
        local overlap = utils.interval_overlap(segs:index(1, I), segs[i]:resize(1, 2))
        I = I[overlap:le(thresh)]
    end

    pick = pick[{{1, counter-1}}]
    return pick
end


function utils.getTube(vid_name, loc, tube_size)
	assert(vid_name, 'Argument #1 is needed to specify video.')
	assert(tube_size and torch.type(tube_size) == 'number')
    rootPath = "/media/caffe/wjz1/data/Videos/activitynet/data/"
    print("vidname: ", vid_name)
	local vid = torchvid.Video.new(rootPath .. vid_name)
    local count = vid:get_image_frame_count()
    print("count: ", count)
    print("loc: ", loc)
    local frame_number, _ = utils.get_bound_frame(count, loc, tube_size)
    print("frame_number: ", frame_number)
	local fps = vid:guess_image_frame_rate()
    print("fps: ", fps)
	local seek_point = (frame_number - 1) / fps
    print(vid:duration())
	print("seekpoit: ", seek_point)
	local key_frame = vid:seek(seek_point)
	target_tube = torch.Tensor(3, tube_size, 224, 224)
	for i = 1, tube_size do
		local frame = key_frame:next_image_frame():to_float_tensor()
		frame_resize = image.scale( frame, 224, 224 ):resize( 3, 1, 224, 224)
		target_tube[{{},{i},{},{}}]:copy(frame_resize)
	end
	return target_tube
end

function utils.get_bound_frame(duration, l, tube_size)
    local frameIdx = math.floor(l * duration)
    local stFrame, edFrame = frameIdx-tube_size/2+1, frameIdx+tube_size/2
    if stFrame < 1 then
       stFrame, edFrame = 1, tube_size
    end
    if edFrame > duration then
        stFrame, edFrame = duration-tube_size+1, duration
    end
    return stFrame, edFrame
end

function utils.Set(list)
    local set = {}
    for _, l in ipairs(list) do set[l] = true end
    return set
end

return utils
