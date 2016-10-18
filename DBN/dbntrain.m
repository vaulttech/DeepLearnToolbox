function dbn = dbntrain(dbn, x, opts)
    n = numel(dbn.rbm);

    dbn.rbm{1} = rbmtrain(dbn.rbm{1}, x, opts, 1);
    for i = 2 : n
        x = rbmup(dbn.rbm{i - 1}, x);
        tic;
        dbn.rbm{i} = rbmtrain(dbn.rbm{i}, x, opts, i);
        toc;
    end

end
