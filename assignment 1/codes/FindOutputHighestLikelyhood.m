%Take a set of results of estimation and from it create an array of
%estimates in which most frequent class choice for every index is used as
%the estimated class
resultsarray=zeros(1,length(estimatesunkowns(1,:)));
for a=1:length(estimatesunkowns(1,:))
    refarray =zeros(1,7);
    for b=1:length(estimatesunkowns(:,1))
        refarray(estimatesunkowns(b,a))=refarray(estimatesunkowns(b,a))+1;
    end
    [~,resultsarray(a)]=max(refarray);
end