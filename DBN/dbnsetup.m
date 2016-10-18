function dbn = dbnsetup(dbn, x, opts)
    % alpha: learning rate
    % momentum: how much the last update influences the current one
    % lambda: sparsity weight

    n = size(x, 2);
    dbn.sizes = [n, dbn.sizes];

    for u = 1 : numel(dbn.sizes) - 1
        dbn.rbm{u}.alpha    = opts.alpha;
        dbn.rbm{u}.momentum = opts.momentum;

        dbn.rbm{u}.W  = rand(dbn.sizes(u + 1), dbn.sizes(u));
        dbn.rbm{u}.vW = rand(dbn.sizes(u + 1), dbn.sizes(u));

        dbn.rbm{u}.b  = rand(dbn.sizes(u), 1);
        dbn.rbm{u}.vb = rand(dbn.sizes(u), 1);

        dbn.rbm{u}.c  = rand(dbn.sizes(u + 1), 1);
        dbn.rbm{u}.vc = rand(dbn.sizes(u + 1), 1);
    end

end
