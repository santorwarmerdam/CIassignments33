function [ outputvec ] = neuronoutputs( weights, inputs,threshold )
%Simply calculate the output of the next layer of neurons given weigths
%matrix, threshold and inputs to the layer
outputvec=weights*inputs;
temp=ones(length(threshold),1);
outputvec=temp./(temp+exp(-(outputvec-threshold)));
end

