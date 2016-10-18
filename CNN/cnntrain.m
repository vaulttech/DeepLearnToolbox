function net = cnntrain(net, x, y, opts)
    % Gets the number of batches
    m = size(x, opts.dims + 1);
    numbatches = m;
    if opts.batchsize > 1
        numbatches = m / opts.batchsize;
    end

    if rem(numbatches, 1) ~= 0
        warning('numbatches not integer: discarding last batch, of irregular size');
        numbatches = floor(numbatches);
    end

    net.rL = [];
    for i = 1 : opts.numepochs
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)]);
        tic;
        kk = randperm(m);
        for l = 1 : numbatches
            if (mod(l, 100) == 0)
                % Print the current batch after each 100 batches are gone
                fprintf('batch %d/%d\t', l, numbatches);
            end

            if (opts.dims == 2)
                batch_x = x(:, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
                batch_y = y(:,    kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            elseif (opts.dims == 1)
                batch_x = x(:,  kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
                batch_y = y(:,  kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            end

            net = cnnff(net, batch_x, opts);
            net = cnnbp(net, batch_y, opts);
            net = cnnapplygrads(net, opts);

            if isempty(net.rL)
                net.rL(1) = net.L;
            end
            net.rL(end + 1) = 0.99 * net.rL(end) + 0.01 * net.L;
        end
        toc;
    end
    
end
