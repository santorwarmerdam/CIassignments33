function [ classifications ] = main( features, targets, unknown)
    NumberInitializations=1;%Number of different initil
    %Parameters to be varied and tested:
    testsize=1000;%Size of validation set
    validationsize=1000;%Size of test set
    inputneurons=10;
    outputneurons=7;
    hiddenneurons=25;%Test how changing this affects error
    learningrate=0.1;%Test how changing this affects error over time
    maxepochs=80;%maximum number of epochs for training, should never actually be reached
    momentum=0;%amount of momentum 0.95 standard if used, 0 if you don't want to use momentum
    acceleration=1;%Learning rate change if going downhill 1.05 standard, 1 if you don't want to use acceleration
    deceleration=1;%learning rate change if not going downhill 0.7 standard,1 if you don't want to use deceleration
    downhilratio=1;%if newest sum square error is this much lower than previous increase learningrate
    uphilratio=1.04;%if newest sum square error is this much higher than previous decrease learningrate

    targetsvec=zeros(outputneurons,length(targets));
    for i=1:length(targets)%Turning class numbers into an desired output vector of the network
        targetsvec(targets(i),i)=1;
    end

    sumerrortraining=zeros(1,maxepochs);
    estimatesunkowns=zeros(NumberInitializations,length(unknown));
    testestimations=zeros(NumberInitializations,testsize);%For all initalisation+training sets determine eventual test estimations, to check expected correct%    
    for initializationnumber=1:NumberInitializations
        %Initializations
        weightsIH=(rand(hiddenneurons,inputneurons)-0.5)*(4.8/inputneurons);
        weightsHO=(rand(outputneurons,hiddenneurons)-0.5)*(4.8/hiddenneurons);
        changeweightsIH=zeros(hiddenneurons,inputneurons);
        changeweightsHO=zeros(outputneurons,hiddenneurons);
        thresholdsH=(rand(1,hiddenneurons)-0.5)*(4.8/inputneurons);
        thresholdsO=(rand(1,outputneurons)-0.5)*(4.8/hiddenneurons);
        changethresholdsH=zeros(1,hiddenneurons);
        changethresholdsO=zeros(1,outputneurons);
        outputshidden=zeros(1,hiddenneurons);
        outputsend=zeros(1,outputneurons);
        error=zeros(outputneurons,1);
        sumerror=zeros(1,maxepochs);
        sqsumerrorprev=500;%initialisating of previous error for learningrate

        %Initializations of variables used only during tuning of algorithm
        weightsIHstart=zeros(hiddenneurons,inputneurons);
        weightsHOstart=zeros(outputneurons,hiddenneurons);
        changeweightsIHoverall=zeros(maxepochs,hiddenneurons,inputneurons);
        changeweightsHOoverall=zeros(maxepochs,outputneurons,hiddenneurons);
        thresholdsHstart=zeros(1,hiddenneurons);
        thresholdsOstart=zeros(1,outputneurons);
        changethresholdsHoverall=zeros(maxepochs,hiddenneurons);
        changethresholdsOoverall=zeros(maxepochs,outputneurons);
        countincorrect=zeros(1,maxepochs);
        learningratelog = zeros(1,maxepochs);
        stop=0;
        epoch=0;
        %Training
        while(stop==0)
            epoch=epoch+1;
            %Store current values of all weights and tresholds so we can
            %determine total change in each epoch, which is used in optimizing
            %the program
            weightsIHstart=weightsIH;
            weightsHOstart=weightsHO;
            thresholdsHstart=thresholdsH;
            thresholdsOstart=thresholdsO;
            for a=1:(length(features)-testsize-validationsize)%1 Epoch of training
                %First find outputs of the complete network
                outputshidden=neuronoutputs( weightsIH, features(a,:)',thresholdsH');
                outputsend=neuronoutputs( weightsHO, outputshidden,thresholdsO');

                error=targetsvec(:,a)-outputsend;%Error in the outputs 
                deltak=(outputsend.*(ones(7,1)-outputsend)).*(error);%determine delta for output and hidden layer for sigma activation function
                deltaj= (deltak'*weightsHO).*((outputshidden.*(ones(hiddenneurons,1)-outputshidden))');

                %Compute change in weights and set new weights
                changeweightsIH=momentum*changeweightsIH+(learningrate*deltaj')*features(a,:);%Weight training from input to hidden layer
                changeweightsHO=momentum*changeweightsHO+learningrate*deltak*outputshidden';%Weight training from hidden to output layer
                weightsIH=weightsIH+changeweightsIH;
                weightsHO=weightsHO+changeweightsHO;
                   
                %Compute change in treshold and set new tresholds
                changethresholdsH=momentum*changethresholdsH-learningrate*deltaj;
                changethresholdsO=momentum*changethresholdsO-learningrate*deltak';
                thresholdsH=thresholdsH+changethresholdsH;
                thresholdsO=thresholdsO+changethresholdsO;
                sumerrortraining(epoch)=sumerrortraining(epoch)+sum((error.^2));
            end
            sumerrortraining(epoch)=sumerrortraining(epoch)/(length(features)-testsize-validationsize);
            %Check against validation set to determine if training end
            %condition has been reached
            for a=(length(features)-testsize-validationsize+1):(length(features)-testsize)
                outputshidden=neuronoutputs( weightsIH, features(a,:)',thresholdsH');
                outputsend=neuronoutputs( weightsHO, outputshidden,thresholdsO');
                error=targetsvec(:,a)-outputsend;%Error of the outputs for validation set
                sumerror(epoch) = sumerror(epoch)+sum((error.^2));%create sum of square errors this epoch
                [~,estout]=max(outputsend); %Take maximum output as the estimated class
                countincorrect(epoch)=countincorrect(epoch)+(estout~=targets(a));%Count number of incorrect class choices this epoch
            end
            sumerror(epoch)=sumerror(epoch)/(validationsize);%Normalise error over epoch to average error per case

            %Store all the total weight changes in this epoch(for
            %optimizing the algorithm)
            changeweightsIHoverall(epoch,:,:)=weightsIH-weightsIHstart;
            changeweightsHOoverall(epoch,:,:)=weightsHO-weightsHOstart;
            changethresholdsHoverall(epoch,:)=thresholdsHstart-thresholdsH;
            changethresholdsOoverall(epoch,:)=thresholdsOstart-thresholdsO;

            %Learning rate adjustments
            if((uphilratio*sqsumerrorprev)<(sumerror(epoch)))
                learningrate=learningrate*deceleration;
            elseif((downhilratio*sqsumerrorprev)>(sumerror(epoch)))
                 learningrate=learningrate*acceleration;
            end
            %Store learningrates over different epochs, used to optimize
            %algorithm
            learningratelog(epoch)=learningrate;
            sqsumerrorprev=sumerror(epoch);
            
            
            if((epoch>5))%Check if the training criteria has been reached (a change of less than 1% over 5 epochs)
                if(or(sumerror(epoch)>0.99*sumerror(epoch-5),epoch==maxepochs))
                    stop=1;
                end
            elseif(epoch==maxepochs)
                stop=1;
            end
        end

        %Checking against known test set
        f=0;
        correct=0;
        sumerrorsq=zeros(1,testsize);
        for a=(length(features)-(testsize-1)):length(features)
            f=f+1;
            outputshidden=neuronoutputs( weightsIH, features(a,:)',thresholdsH');
            outputsend=neuronoutputs( weightsHO, outputshidden,thresholdsO');
            [~,neuron]=max(outputsend);        
            testestimations(initializationnumber,f)=neuron;
            if(neuron==targets(a))%Find number of correct classifications in the test set
                correct=correct+1; 
            end
            for b=1:outputneurons
                sumerrorsq(f)=sumerrorsq(f)+((targets(a)==b)-outputsend(b))^2;
            end
        end
        
        for a=1:length(unknown)%Classify the unkown set
            outputshidden=neuronoutputs(weightsIH, unknown(a,:)',thresholdsH');
            outputsend=neuronoutputs( weightsHO, outputshidden,thresholdsO');
            [~,neuron]=max(outputsend);
            estimatesunkowns(initializationnumber,a)=neuron;
        end

    end
    
    %Take the set of results of estimation and from it create an array of
    %estimates in which most frequent class choice for every index is used as
    %the estimated class
    classifications=zeros(1,length(estimatesunkowns(1,:)));
    for a=1:length(estimatesunkowns(1,:))
        refarray =zeros(1,7);
        for b=1:length(estimatesunkowns(:,1))
            refarray(estimatesunkowns(b,a))=refarray(estimatesunkowns(b,a))+1;
        end
        [~,classifications(a)]=max(refarray);
    end
end