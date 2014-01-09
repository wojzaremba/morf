function [incr, err] = Test(train)
global plan;
plan.training = 0;
incr = 0;
all = 0;
input = plan.input;
input.step = 1;
fprintf('Testing:\n');
while (true)
    input.GetImage(train);
    if (input.step == -1)
        break;
    end
    fprintf('*');
    ForwardPass(plan.input);    
    incr = incr + plan.classifier.GetScore();
    all = all + plan.input.batch_size;
    err = sum(incr) / all;
end
plan.training = 1;
end
