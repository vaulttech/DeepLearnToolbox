function net = cnnff(net, x, opts)
    n = numel(net.layers);
    net.layers{1}.a{1} = x;
    inputmaps = 1;

    for l = 2 : n   %  for each layer
        if strcmp(net.layers{l}.type, 'c')
            %  !!below can probably be handled by insane matrix operations
            for j = 1 : net.layers{l}.outputmaps   %  for each output map
                %  create temp output map
                %z = zeros(size(net.layers{l - 1}.a{1}) - [net.layers{l}.kernelsize(1) - 1 net.layers{l}.kernelsize(2) - 1]);
                z = create_temp_out_map(net, l, opts);
                for i = 1 : inputmaps   %  for each input map
                    %  convolve with corresponding kernel and add to temp output map
                    z = z + convn(net.layers{l - 1}.a{i}, net.layers{l}.k{i}{j}, 'valid');
                end
                %  add bias, pass through nonlinearity
                net.layers{l}.a{j} = sigm(z + net.layers{l}.b{j});
            end
            %  set number of input maps to this layers number of outputmaps
            inputmaps = net.layers{l}.outputmaps;
        elseif strcmp(net.layers{l}.type, 's')
            %  downsample
            for j = 1 : inputmaps
                % This is a very strange way of downsampling O__O
                scale_vector = net.layers{l}.scale;
                if (opts.dims == 1)
                    scale_vector = [net.layers{l}.scale 1];
                end
                z = convn(net.layers{l - 1}.a{j}, ones(scale_vector) / prod(net.layers{l}.scale), 'valid');   %  !! replace with variable
                %z = conv(net.layers{l - 1}.a{j}, ones(net.layers{l}.scale)' ./ (net.layers{l}.scale), 'valid');   %  !! replace with variable

                % TODO: find a way to make this generic for any number of
                % dimentions
                if opts.dims == 1
                    net.layers{l}.a{j} = z(1 : net.layers{l}.scale(1) : end, :);
                elseif opts.dims == 2
                    net.layers{l}.a{j} = z(1 : net.layers{l}.scale(1) : end, 1 : net.layers{l}.scale(2) : end, :);
                end
            end
        end
    end

    %  concatenate all end layer feature maps into vector
    %  (fv = feature vector)
    net.fv = [];
    for j = 1 : numel(net.layers{n}.a)
        sa = size(net.layers{n}.a{j});
        dimensions = opts.dims+1;
        if opts.batchsize == 1
            % Artificially adds 1 dimension (Matlab discards the last
            % dimension if the batchsize is 1)
            sa(dimensions) = 1;
        end
        net.fv = [net.fv; reshape(net.layers{n}.a{j}, prod(sa(1 : dimensions-1)), sa(dimensions))];
    end

    %  feedforward into output perceptrons
    net.o = net.ffW * net.fv + repmat(net.ffb, 1, size(net.fv, 2));

    % If I am not doing regression, then I want to apply the sigmoid
    % function
    apply_sigmoid = ~(isfield(opts, 'regression') && opts.regression == 1);
    if(apply_sigmoid)
        net.o = sigm(net.o);
    end

end

function ret = create_temp_out_map(net, l, opts)
    m = [];
    for k = 1:opts.dims
        m(k) = net.layers{l}.kernelsize(k)-1;
    end
    if opts.batchsize > 1 || opts.dims == 1
        m(opts.dims+1) = 0;
    end
    ret = zeros(size(net.layers{l - 1}.a{1}) - m);
end
