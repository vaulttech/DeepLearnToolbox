function net = cnnbp(net, y, opts)
    n = numel(net.layers);

    %   error
    net.e = net.o - y;
    %  loss function
    net.L = 1/2* sum(net.e(:) .^ 2) / size(net.e, 2);
    %fprintf('Loss: %f\n', net.L);

    %%  backprop deltas

    %  output delta
    net.od = net.e;

    % If I am not doing regression, then the feedforward function applied
    % a sigmoid function in the end of the output layer, and the derivative
    % of the loss should include the derivative of the sigmoid (which is
    % calculated as `sigm(x) * (1 - sigm(x)`).
    applied_sigmoid = ~(isfield(opts, 'regression') && opts.regression == 1);
    if (applied_sigmoid)
        net.od = net.od .* (net.o .* (1 - net.o));
    end

    %  feature vector delta
    net.fvd = (net.ffW' * net.od);

    %  only conv layers has sigm function
    if strcmp(net.layers{n}.type, 'c')
        net.fvd = net.fvd .* (net.fv .* (1 - net.fv));
    end

    %  reshape feature vector deltas into output map style
    sa = size(net.layers{n}.a{1});
    dimensions = opts.dims;
    if opts.batchsize == 1
        % Artificially adds 1 dimension (Matlab discards the last
        % dimension if the batchsize is 1)
        dimensions = dimensions + 1;
        sa(dimensions) = 1;
    end
    fvnum = prod(sa(1:dimensions));
    for j = 1 : numel(net.layers{n}.a)
        net.layers{n}.d{j} = reshape(net.fvd(((j - 1) * fvnum + 1) : j * fvnum, :), sa);
        %net.layers{n}.d{j} = reshape(net.fvd(((j - 1) * fvnum + 1) : j * fvnum, :), sa(1), sa(2));
    end

    for l = (n - 1) : -1 : 1
        if strcmp(net.layers{l}.type, 'c')
            for j = 1 : numel(net.layers{l}.a)
                sa = [net.layers{l + 1}.scale];
                if (opts.batchsize > 1 || opts.dims == 1)
                    sa(end+1) = 1;
                end
                net.layers{l}.d{j} = net.layers{l}.a{j} .* ...
                    (1 - net.layers{l}.a{j}) .* ...
                    (expand(net.layers{l + 1}.d{j}, sa)/ prod(net.layers{l + 1}.scale));
                %net.layers{l}.d{j} = net.layers{l}.a{j} .* (1 - net.layers{l}.a{j}) .* (expand(net.layers{l + 1}.d{j}, net.layers{l + 1}.scale) / (net.layers{l + 1}.scale .^ 2)');
            end
        elseif strcmp(net.layers{l}.type, 's')
            for i = 1 : numel(net.layers{l}.a)
                z = zeros(size(net.layers{l}.a{1}));
                for j = 1 : numel(net.layers{l + 1}.a)
                     z = z + convn(net.layers{l + 1}.d{j}, rot180(net.layers{l + 1}.k{i}{j}), 'full');
                     %z = z + conv(net.layers{l + 1}.d{j}, rot180(net.layers{l + 1}.k{i}{j}), 'full');
                end
                net.layers{l}.d{i} = z;
            end
        end
    end

    %%  calc gradients
    for l = 2 : n
        if strcmp(net.layers{l}.type, 'c')
            for j = 1 : numel(net.layers{l}.a)
                for i = 1 : numel(net.layers{l - 1}.a)
                    net.layers{l}.dk{i}{j} = convn(flipall(net.layers{l - 1}.a{i}), net.layers{l}.d{j}, 'valid') / size(net.layers{l}.d{j}, 3);
                end
                net.layers{l}.db{j} = sum(net.layers{l}.d{j}(:)) / size(net.layers{l}.d{j}, 3);
            end
        end
    end
    net.dffW = net.od * (net.fv)' / size(net.od, 2);
    net.dffb = mean(net.od, 2);

    function X = rot180(X)
        X = flipdim(flipdim(X, 1), 2);
    end
end
