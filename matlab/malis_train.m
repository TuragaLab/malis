function m = malis_train(m,show_plot,platform,nice)

% load the network if passed a filename
if(ischar(m))
	load(m,'m');
end


%% -----------------------Initializing
% configure IO
% configure save directory
if(isfield(m.params,'save_directory') && exist(m.params.save_directory,'dir'))
	m.params.save_string=[m.params.save_directory, num2str(m.ID),'/'];
	log_message([],['saving network to directory: ', m.params.save_string]); 
	mkdir(m.params.save_string);
else
	error('save directory not found!');
end


% some minimax params
if ~isfield(m.params,'constrained_minimax'),
	m.params.constrained_minimax = true;
end

% initialize statistics
% initialize clocks
if(~isfield(m.stats, 'last_backup'))
	m.stats.last_backup=clock;
end

maxEpoch = ceil(m.params.maxIter/m.params.nIterPerEpoch);
if ~isfield(m.stats,'epoch_loss'),
	m.stats.epoch_loss = zeros(m.params.nIterPerEpoch,1,'single');
	m.stats.epoch_classerr = zeros(m.params.nIterPerEpoch,1,'single');
	m.stats.epoch_randIndex = zeros(m.params.nIterPerEpoch,1,'single');
else,
	m.stats.epoch_loss(m.params.nIterPerEpoch) = 0;
	m.stats.epoch_classerr(m.params.nIterPerEpoch) = 0;
	m.stats.epoch_randIndex(m.params.nIterPerEpoch) = 0;
end
if ~isfield(m.stats,'loss'),
	m.stats.loss = zeros(maxEpoch,1,'single');
	m.stats.classerr = zeros(maxEpoch,1,'single');
	m.stats.randIndex = zeros(maxEpoch,1,'single');
	m.stats.times = zeros(maxEpoch,1,'single');    
else,
	m.stats.loss(m.stats.iter+1:maxEpoch) = 0;
	m.stats.classerr(m.stats.iter+1:maxEpoch) = 0;
	m.stats.randIndex(m.stats.iter+1:maxEpoch) = 0;
end

% initialize random number generators
rand('state',sum(100*clock));
randn('state',sum(100*clock));

% init counters
if ~isfield(m.stats,'iter') || m.stats.iter < 1,
	m.stats.iter = 1;
end
if ~isfield(m.stats,'epoch') || m.stats.epoch < 1,
	m.stats.epoch = 1;
end
cnpkg_log_message(m, ['initializing stats']);

% construct cns model for gpu
m = cnpkg4.MapDimFromOutput(m,m.params.graph_size,2*m.params.minibatch_size);
[m,lastStep] = cnpkg4.SetupStepNo(m,1);
% totalBorder = 2*m.offset + cell2mat(m.layers{m.layer_map.output}.size(2:4)) - 1;
% halfTotalBorder = ceil(totalBorder/2);
totalBorder = cell2mat(m.layers{m.layer_map.input}.size(2:4)) - cell2mat(m.layers{m.layer_map.output}.size(2:4));
leftBorder = m.offset;
rightBorder = totalBorder - leftBorder;
halfOutputSize = floor(cell2mat(m.layers{m.layer_map.output}.size(2:4))/2);

m.layers{m.layer_map.minibatch_index}.size = {5,1,m.params.minibatch_size*2};
index = zeros(cell2mat(m.layers{m.layer_map.minibatch_index}.size));
m.layers{m.layer_map.minibatch_index}.val = index;

% load data
data = load(m.data_info.training_files{1});
log_message(m,['Loaded data file...']);

% reformat 'im', apply mask, compute conn
transform_and_subsample_data;
m.inputblock = im;
m.labelblock={[]};
m.maskblock={[]};


fprintf('Initializing on device...'),tic
cns('init',m,platform,nice)
fprintf(' done. '),toc


% done initializing!
log_message(m, ['initialization complete!']);

% if epoch 0, save
if m.stats.epoch == 0,
	m = cns('update',m); m.inputblock = {[]}; m.labelblock = {[]}; m.maskblock = {[]};
    for k=1:length(m.layers),
        switch m.layers{k}.type,
        case {'input', 'hidden', 'output', 'label'}
            if isfield(m.layers{k},'val'), m.layers{k}.val=0; end
            if isfield(m.layers{k},'sens'), m.layers{k}.sens=0; end
        end
    end
	save([m.params.save_string,'epoch-' num2str(m.stats.epoch)],'m');
	save([m.params.save_string,'latest'],'m');
	m.stats.last_backup = clock;
end


%% ------------------------Training
log_message(m, ['beginning training.. ']);
dloss = zeros(cell2mat(m.layers{m.layer_map.error}.size),'single');
while(m.stats.iter < m.params.maxIter),


    %% Timing -----------------------------------------------------------------------------------
    epoch_clock = clock;

	%% Assemble a minibatch ---------------------------------------------------------------------
	%% select a nice juicy image and image patch

	% Pick big patch for negative examples
	imPickNeg = randsample(length(im),1);
	segPickNeg = randsample(length(seg{imPickNeg}),1);

	for k=1:3,
		index(k,1,1) = randi([max(1,bb{imPickNeg}{segPickNeg}(k,1)-m.leftBorder(k)) ...
						min(imSz{imPickNeg}(k),bb{imPickNeg}{segPickNeg}(k,2)+m.rightBorder(k))-m.totalBorder(k)-m.params.graph_size(k)+1],1);
	end
	index(4,1,1) = imPickNeg;
	for k=1:3,
		idxGraphOutNeg{k} = (index(k,1,1)+m.leftBorder(k)-1)+(1:m.params.graph_size(k));
	end
	segNeg = single(connectedComponents( ...
				MakeConnLabel( ...
					seg{imPickNeg}{segPickNeg}(idxGraphOutNeg{1},idxGraphOutNeg{2},idxGraphOutNeg{3}), ...
					m.params.nhood),m.params.nhood));
	maskNeg = mask{imPickNeg}{segPickNeg}(idxGraphOutNeg{1},idxGraphOutNeg{2},idxGraphOutNeg{3});
	cmpsNeg = unique(segNeg.*maskNeg);
	cmpsNeg = cmpsNeg(cmpsNeg~=0);
	% reject patch if it doesn't have more than one object!
	if length(cmpsNeg)<2,
		if m.params.debug, disp('Negative patch doesn''t have more than one object!'), end
		continue
	end
	connNeg = MakeConnLabel(segNeg.*maskNeg,m.params.nhood);
	


	% Pick big patch for positive examples
	imPickPos = randsample(length(im),1);
	segPickPos = randsample(length(seg{imPickPos}),1);

	for k=1:3,
		index(k,1,2) = randi([max(1,bb{imPickPos}{segPickPos}(k,1)-m.leftBorder(k)) ...
						min(imSz{imPickPos}(k),bb{imPickPos}{segPickPos}(k,2)+m.rightBorder(k))-m.totalBorder(k)-m.params.graph_size(k)+1],1);
	end
	index(4,1,2) = imPickPos;
	for k=1:3,
		idxGraphOutPos{k} = (index(k,1,2)+m.leftBorder(k)-1)+(1:m.params.graph_size(k));
	end
	segPos = single(connectedComponents( ...
				MakeConnLabel( ...
					seg{imPickPos}{segPickPos}(idxGraphOutPos{1},idxGraphOutPos{2},idxGraphOutPos{3}), ...
					m.params.nhood),m.params.nhood));
	maskPos = mask{imPickPos}{segPickPos}(idxGraphOutPos{1},idxGraphOutPos{2},idxGraphOutPos{3});
	cmpsPos = unique(segPos.*maskPos);
	cmpsPos = cmpsPos(cmpsPos~=0);
	% reject patch if it doesn't have even one object!
	if length(cmpsPos)<1,
		if m.params.debug,disp('Positive patch doesn''t have even one object!'), end
		continue
	end
	connPos = MakeConnLabel(segPos.*maskPos,m.params.nhood);



	%% Run the fwd pass & get the output ---------------------------------------------------------
% fprintf('Running big fwd pass on gpu...'),tic
	cns('set',{m.layer_map.minibatch_index,'val',index-1});
    connEst = cns('step',[1 m.layers{m.layer_map.output}.stepNo(end)],{m.layer_map.output,'val'});
	connEst = permute(connEst,[2 3 4 1 5]);
% fprintf(' done. '),toc
% fprintf('everything in between...'),tic



	%% Compute MALIS loss -------------------------------------------------------
    [dlossNeg,loss(1),classerr(1),randIndex(1)] = malis_loss_mex( ...
                                                single(connEst(:,:,:,:,1)), ...
                                                m.params.nhood, ...
                                                uint16(segNeg), ...
                                                m.layers{m.layer_map.error}.param, ...
                                                false);
    [dlossPos,loss(2),classerr(2),randIndex(2)] = malis_loss_mex( ...
                                                single(connEst(:,:,:,:,2)), ...
                                                m.params.nhood, ...
                                                uint16(segPos), ...
                                                m.layers{m.layer_map.error}.param, ...
                                                true);
    dloss(:,:,:,:,1) = permute(dlossNeg,[4 1 2 3]);
    dloss(:,:,:,:,2) = permute(dlossPos,[4 1 2 3]);



	%% Run the bkwd pass & update weights ---------------------------------------------------------
% fprintf('done. '), toc
% fprintf('Running backward pass on gpu...'),tic
    cns('set',{m.layer_map.error,'val',dloss});
    cns('step',[m.layers{m.layer_map.error}.stepNo+1 lastStep]);
% fprintf('done. '),toc



	%% Record error statistics --------------------------------------------------------------------
	if m.params.debug >= 2,
		log_message(m,['DEBUG_MODE: loss: ' num2str(mean(loss(:)))])
	end
	m.stats.epoch_iter = rem(m.stats.iter,m.params.nIterPerEpoch)+1;
	m.stats.epoch_loss(m.stats.epoch_iter) = mean(loss(:));
	m.stats.epoch_classerr(m.stats.epoch_iter) = mean(classerr(:));
	m.stats.epoch_randIndex(m.stats.epoch_iter) = mean(randIndex(:));
	m.stats.loss(m.stats.epoch) = mean(m.stats.epoch_loss(1:m.stats.epoch_iter));
	m.stats.classerr(m.stats.epoch) = mean(m.stats.epoch_classerr(1:m.stats.epoch_iter));
	m.stats.randIndex(m.stats.epoch) = mean(m.stats.epoch_randIndex(1:m.stats.epoch_iter));
    m.stats.times(m.stats.epoch) = etime(clock,epoch_clock);    

	%% Save current state ----------------------------------------------------------------------
	if (etime(clock,m.stats.last_backup)>m.params.backup_interval),
		log_message(m, ['Saving network state... Iter: ' num2str(m.stats.iter)]);
		m = cns('update',m);
		m.inputblock = {[]};
		for k=1:length(m.layers),
			switch m.layers{k}.type,
			case {'input', 'hidden', 'output', 'label'}
				if isfield(m.layers{k},'val'), m.layers{k}.val=0; end
				if isfield(m.layers{k},'sens'), m.layers{k}.sens=0; end
			end
		end
		save([m.params.save_string,'latest'],'m');
		m.stats.last_backup = clock;
	end

	%% Update counters --------------------------------------------------------------------------
	m.stats.iter = m.stats.iter+1;

	%% Compute test/train statistics ------------------------------------------------------------
	if ~rem(m.stats.iter,m.params.nIterPerEpoch*m.params.nEpochPerSave),
		%% save current state/statistics
		m = cns('update',m);
		m.inputblock = {[]};
		for k=1:length(m.layers),
			switch m.layers{k}.type,
			case {'input', 'hidden', 'output', 'label'}
				if isfield(m.layers{k},'val'), m.layers{k}.val=0; end
				if isfield(m.layers{k},'sens'), m.layers{k}.sens=0; end
			end
		end
		save([m.params.save_string,'epoch-' num2str(m.stats.epoch)],'m');
		save([m.params.save_string,'latest'],'m');
		%% new epoch
		log_message(m,['Epoch: ' num2str(m.stats.epoch) ', Iter: ' num2str(m.stats.iter) '; Classification error: ' num2str(m.stats.classerr(m.stats.epoch))]);
	end
	if ~rem(m.stats.iter,m.params.nIterPerEpoch),
		m.stats.epoch = m.stats.epoch+1;

		%% Reload data every so often --------------------------------------------------------------
		if ~rem(m.stats.epoch,m.params.nEpochPerDataBlock),
			transform_and_subsample_data;
			for mvIdx = 1:m.params.nDataBlock,
				cns('set',{0,'inputblock',mvIdx,im{mvIdx}});
			end
		end
	end

	%% Plot statistics ------------------------------------------------------------
	try,
		if ~rem(m.stats.iter,2e1),
			disp(['Loss(iter: ' num2str(m.stats.iter) ') ' num2str(m.stats.epoch_loss(m.stats.epoch_iter)) ', classerr: ' num2str(m.stats.epoch_classerr(m.stats.epoch_iter))])
			figure(10)
			subplot(121)
			plot(1:m.stats.epoch,m.stats.loss(1:m.stats.epoch))
			subplot(122)
			plot(1:m.stats.epoch,m.stats.classerr(1:m.stats.epoch))
			drawnow
		end
	catch,
	end

end


function transform_and_subsample_data
log_message(m,['Assembling dataBlock']);

for imIdx = 1:m.params.nDataBlock,
	imPick = randsample(length(data.im),1);
	segPick = randsample(length(data.seg{imPick}),1);
	for k=1:3,
		stidx = randi([data.bb{imPick}{segPick}(k,1) ...
							max(data.bb{imPick}{segPick}(k,1), ...
								data.bb{imPick}{segPick}(k,2)-m.params.dataBlockSize(k))],1);
		idx{k} = stidx+(0:m.params.dataBlockSize(k)-1);
	end
	im{imIdx} = data.im{imPick}(idx{1},idx{2},idx{3},:);
	seg{imIdx} = {}; mask{imIdx} = {}; bb{imIdx} = {};	
	for segIdx = 1:length(data.seg{imPick}),
		seg{imIdx}{segIdx} = data.seg{imPick}{segIdx}(idx{1},idx{2},idx{3});
		mask{imIdx}{segIdx} = data.mask{imPick}{segIdx}(idx{1},idx{2},idx{3});
		for k=1:3,
			bb{imIdx}{segIdx}(k,:) = data.bb{imPick}{segIdx}(k,:)-idx{k}(1)+1;
			bb{imIdx}{segIdx}(k,2) = min(bb{imIdx}{segIdx}(k,2),m.params.dataBlockSize(k));
			imSz{imIdx}(k) = length(idx{k});
		end
	end


	% transform
	flp1 = (rand>.5)&m.params.dataBlockTransformFlp(1);
	flp2 = (rand>.5)&m.params.dataBlockTransformFlp(2);
	flp3 = (rand>.5)&m.params.dataBlockTransformFlp(3);
	prmt = (rand>.5)&m.params.dataBlockTransformPrmt(1);

	if prmt,
		im{imIdx} = permute(im{imIdx},[2 1 3 4]);
		for segIdx = 1:length(seg{imIdx}),
			seg{imIdx}{segIdx} = permute(seg{imIdx}{segIdx},[2 1 3 4]);
			mask{imIdx}{segIdx} = permute(mask{imIdx}{segIdx},[2 1 3 4]);
			bb{imIdx}{segIdx} = bb{imIdx}{segIdx}([2 1 3],:);
		end
	end

	if flp1,
		im{imIdx} = flipdim(im{imIdx},1);
		for segIdx = 1:length(seg{imIdx}),
			seg{imIdx}{segIdx} = flipdim(seg{imIdx}{segIdx},1);
			mask{imIdx}{segIdx} = flipdim(mask{imIdx}{segIdx},1);
			oldbb = bb{imIdx}{segIdx};
			bb{imIdx}{segIdx}(1,1) = size(im{imIdx},1)-oldbb(1,2)+1;
			bb{imIdx}{segIdx}(1,2) = size(im{imIdx},1)-oldbb(1,1)+1;
		end
	end

	if flp2,
		im{imIdx} = flipdim(im{imIdx},2);
		for segIdx = 1:length(seg{imIdx}),
			seg{imIdx}{segIdx} = flipdim(seg{imIdx}{segIdx},2);
			mask{imIdx}{segIdx} = flipdim(mask{imIdx}{segIdx},2);
			oldbb = bb{imIdx}{segIdx};
			bb{imIdx}{segIdx}(2,1) = size(im{imIdx},2)-oldbb(2,2)+1;
			bb{imIdx}{segIdx}(2,2) = size(im{imIdx},2)-oldbb(2,1)+1;
		end
	end

	if flp3,
		im{imIdx} = flipdim(im{imIdx},3);
		for segIdx = 1:length(seg{imIdx}),
			seg{imIdx}{segIdx} = flipdim(seg{imIdx}{segIdx},3);
			mask{imIdx}{segIdx} = flipdim(mask{imIdx}{segIdx},3);
			oldbb = bb{imIdx}{segIdx};
			bb{imIdx}{segIdx}(3,1) = size(im{imIdx},3)-oldbb(3,2)+1;
			bb{imIdx}{segIdx}(3,2) = size(im{imIdx},3)-oldbb(3,1)+1;
		end
	end

	im{imIdx} = permute(im{imIdx},[4 1 2 3]);
end

end


end
