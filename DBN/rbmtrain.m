function rbm = rbmtrain(rbm, x, opts, curr_layer_index)
    assert(isfloat(x), 'x must be a float');
    assert(all(x(:)>=0) && all(x(:)<=1), 'all data in x must be in [0:1]');
    m = size(x, 1);
    numbatches = m / opts.batchsize;
    
    assert(rem(numbatches, 1) == 0, 'numbatches not integer');

    for i = 1 : opts.numepochs
        kk = randperm(m);
        err = 0;
        for l = 1 : numbatches
            batch = x(kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize), :);
            
            v1 = batch;

            % I need the probabilities for the sparsity calculation later
            h1_prob = sigm(repmat(rbm.c', opts.batchsize, 1) + v1 * rbm.W');
            h1 = h1_prob > rand(size(h1_prob));

            v2 = sigmrnd(repmat(rbm.b', opts.batchsize, 1) + h1 * rbm.W);
            h2 = sigm(repmat(rbm.c', opts.batchsize, 1) + v2 * rbm.W');

            c1 = h1' * v1;
            c2 = h2' * v2;

            if i < 5
                momentum = rbm.momentum(1);
            else
                momentum = rbm.momentum(2);
            end

            rbm.vW = momentum * rbm.vW + rbm.alpha * (c1 - c2)     / opts.batchsize;
            rbm.vb = momentum * rbm.vb + rbm.alpha * sum(v1 - v2)' / opts.batchsize;
            rbm.vc = momentum * rbm.vc + rbm.alpha * sum(h1 - h2)' / opts.batchsize;

            rbm.W = rbm.W + rbm.vW;
            rbm.b = rbm.b + rbm.vb;
            rbm.c = rbm.c + rbm.vc;

            if opts.sparsity > 0
                % Adds sparsity to the bias of the hidden layer (i.e., to "c")
                rbm.c = rbm.c - opts.lambda * squeeze(mean(h1_prob, 1)' - opts.sparsity);
            end

            err = err + sum(sum((v1 - v2) .^ 2)) / opts.batchsize;
        end
        
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)  '. Average reconstruction error is: ' num2str(err / numbatches)]);
        if (mod(i,10) == 1)
            figure(curr_layer_index), colormap(gray), display_network(rbm.W);
        end
    end
end

function display_network(W)
    count = 1;
    num_features = size(W, 1);
    size_features = size(W, 2);
    features = reshape(W', sqrt(size_features), sqrt(size_features), num_features);
    num_columns = 10;%num_features / 10;
    num_rows = 11;%num_features / num_columns + 1;
    for j = 1 : 110%num_features
        % Idea for normalization gotten from
        % https://de.mathworks.com/matlabcentral/answers/75568-how-can-i-normalize-data-between-0-and-1-i-want-to-use-logsig
        subplot(num_rows, num_columns, count);
        minVal = min(min(features(:,:,j)));
        maxVal = max(max(features(:,:,j)));
        features2(:,:,j) = (features(:,:,j) - minVal) / (maxVal - minVal);
        imshow(features2(:,:,j)');
        count = count + 1;
    end
    drawnow;
end

