function RunRegular()
global plan;
plan.training = 1;
assert(length(plan.layer) > 1);
input = plan.input;
last_repeat = input.repeat;
costs = {};
for i = 1:length(plan.layer)
    if (length(plan.layer{i}.next) == 0)
        costs{end + 1} = plan.layer{i};
    end
end
for repeat = last_repeat:input.max_repeat
    input.training = 1;
    repeattime = tic;
    incorrect = 0;
    all = 0;
    input.step = 1;
    while (true)
        input.GetImage(1);
        if (input.step == -1)
            break;
        end
        fprintf('*');
        ForwardPass(plan.input);
        for c = 1:length(costs)
            BackwardPass(costs{c});
        end
        incorrect = incorrect + plan.classifier.GetScore();
        all = all + input.batch_size;
        if (mod((input.step - 1), floor(input.train.batches / 5)) == 0)
            fprintf('incorrect = %d, all = %d\n', incorrect, all);
        end
    end
    if (repeat > 10)
        fprintf('Regularizing\n');
        W = plan.layer{2}.cpu.vars.W;
        [U, S, V] = svd(W);
        s = 80;
        plan.layer{2}.cpu.vars.W = U(:, 1:s) * S(1:s, 1:s) * V(:, 1:s)';        
        
        W = plan.layer{3}.cpu.vars.W;
        [U, S, V] = svd(W);
        s = 40;
        plan.layer{3}.cpu.vars.W = U(:, 1:s) * S(1:s, 1:s) * V(:, 1:s)';                
    end
    input.repeat = repeat + 1;
    fprintf('\nEpoch took = %f\n', toc(repeattime));   
    [incr_test, err] = Test(0);
    fprintf('\nepoch = %d, incr_test = %d, err = %f\n', repeat, incr_test, err);
end
end

