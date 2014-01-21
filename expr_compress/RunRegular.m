function RunRegular()
global plan
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
%     if (repeat > 0)
%         fprintf('Regularizing\n');
%         layers = [2, 3];
% %         s = 2;
%         for i = layers
%             [U, S, V] = svd(plan.layer{i}.cpu.vars.W);
% %             plan.layer{i}.cpu.vars.W = U(:, s:end) * S(s:end, s:end) * V(:, s:end)';
%             k = min(size(S));
%             S = diag(S);
%             S = sum(S) .* sqrt(S) ./ (sum(sqrt(S)));
%             S = diag(S);
%             plan.layer{i}.cpu.vars.W = U * S * V(:, 1:k)';
%         end
%     end
    input.repeat = repeat + 1;
    fprintf('\nEpoch took = %f\n', toc(repeattime));   
    [incr_test, err] = Test(0);
    fprintf('\nepoch = %d, incr_test = %d, err = %f\n', repeat, incr_test, err);
    plan.incr_tests = [plan.incr_tests; incr_test];
    plan.incr_trains = [plan.incr_trains; incorrect / 6];
    plot(plan.incr_tests(plan.incr_tests < 300), 'r');
    hold on
    plot(plan.incr_trains(plan.incr_trains < 300), 'b');
    legend('Test data', 'Train data');
    hold off
    drawnow
end
end

