%Take a set of test results of estimation and from it create an array of
%estimates in which most frequent class choice for every index is used as
%the estimated class and check the correct percentage of that array
resultsarray=zeros(1,length(testestimations(1,:)));
for a=1:length(testestimations(1,:))
    refarray =zeros(1,7);
    for b=1:length(testestimations(:,1))
        refarray(testestimations(b,a))=refarray(testestimations(b,a))+1;
    end
    [~,resultsarray(a)]=max(refarray);
end

testcorrectaveraged=0;
for a=1:length(resultsarray)
    f=a+length(features)-length(resultsarray);
    if(resultsarray(a)==targets(f))
        testcorrectaveraged = testcorrectaveraged+1;
    end
end