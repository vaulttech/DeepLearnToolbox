function net = cnnsetup(net, x, y, opts)
    assert(~isOctave() || compare_versions(OCTAVE_VERSION, '3.8.0', '>='), ['Octave 3.8.0 or greater is required for CNNs as there is a bug in convolution in previous versions. See http://savannah.gnu.org/bugs/?39314. Your version is ' myOctaveVersion]);
    inputmaps = 1;
    mapsize = size(x);
    mapsize = mapsize(1:opts.dims);

    for l = 1 : numel(net.layers)   %  layer
        if strcmp(net.layers{l}.type, 's')
            % Scales the map
            %mapsize = mapsize / net.layers{l}.scale;
            mapsize = mapsize ./ net.layers{l}.scale;

            assert(all(floor(mapsize)==mapsize), ['Layer ' num2str(l) ' size must be integer. Actual: ' num2str(mapsize)]);

            % Initializes the biases
            for j = 1 : inputmaps
                net.layers{l}.b{j} = 0;
            end
        end
        if strcmp(net.layers{l}.type, 'c')
            mapsize = mapsize - net.layers{l}.kernelsize + 1;
            fan_out = net.layers{l}.outputmaps * prod(net.layers{l}.kernelsize);
            for j = 1 : net.layers{l}.outputmaps  %  output map
                fan_in = inputmaps * prod(net.layers{l}.kernelsize);
                for i = 1 : inputmaps  %  input map
                    random_weights = net.layers{l}.kernelsize;
                    if (opts.dims == 1)
                        random_weights = [random_weights 1];
                    end
                    net.layers{l}.k{i}{j} = (rand(random_weights) - 0.5) * 2 * sqrt(6 / (fan_in + fan_out));
                end
                net.layers{l}.b{j} = 0;
            end
            inputmaps = net.layers{l}.outputmaps;
        end
    end

    % 'onum' is the number of labels, that's why it is calculated using
    % size(y, 1). If you have 20 labels so the output of the network will
    % be 20 neurons.
    %
    % 'fvnum' is the number of output neurons at the last layer, the layer
    % just before the output layer.
    %
    % 'ffb' is the biases of the output neurons.
    %
    % 'ffW' is the weights between the last layer and the output neurons.
    % Note that the last layer is fully connected to the output layer,
    % that's why the size of the weights is (onum * fvnum)
    fvnum = prod(mapsize) * inputmaps;
    onum = size(y, 1);

    net.ffb = zeros(onum, 1);
    net.ffW = (rand(onum, fvnum) - 0.5) * 2 * sqrt(6 / (onum + fvnum));
end
