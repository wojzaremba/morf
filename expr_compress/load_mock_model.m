function load_mock_model()
    global plan
    jsons = {};
    jsons{1} = struct('batch_size', 100, 'rows', 2, 'cols', 3, 'depth', 4, 'number_of_classes', 10, 'type', 'TestInput');
    jsons{2} = struct('function', 'LINEAR', 'rows', 1, 'cols', 1, 'depth', 10, 'type', 'FC', 'fully_connected', true);    
    jsons{3} = struct('type', 'Softmax');
    plan = Plan(jsons);    
end