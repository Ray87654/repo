import processing.serial.*;

final short select_length = 465;
final short select_width = 100;
final short block_length = 30;
final short block_width = 30;

Serial USB_Port;
int[] Serial_data = new int[3];
int[] data = new int[4];

short mode = 0;
short choose = 0;
short initial_times = 0;
short release = 0;      //for mouse

// for choose 4
float count = 0;
ArrayList<Integer> remain = new ArrayList<>();

board Easy = new board(20, 10, 25, 1.9);
board Normal = new board(30, 15, 60, 2.8);
board Hard = new board(35, 20, 120, 3.5);

public class board
{
    private int board_width, board_height, mines, remaining_mines;
    private float density;
    private int[][] mineboard, covered; // 0:not change, 1~8 :mines quantites, 9:all side clear, 10:Flag, 11:flag > mines
    private int[] n_colors = {#404040, #FFFFA7, #B8FF95, #71FFA1, #55EFFF, #7F90C4, #7845DB, #E08E29, #FA2E1C, #FAFAFA, 100, #C92FF5}; 
                           //    0        1       2         3        4        5        6        7        8        9     10      11

    public void plant_mines(int difficulty)
    {
        float Probability = difficulty * 0.2208;
        ArrayList Used = new ArrayList();
        int current = int(random(0, this.board_width*this.board_height));
        int times = 0;
        int[] mines;
        while(times < this.mines)
        {
            while (Used.contains(current))
            {
                current = int(random(0, this.board_width*this.board_height));  
            }
            mines = Detect_mines(Detect_array_avaliable(current), current/this.board_width, current%this.board_width);
            if(mines[8] == 8)
            {
                mines[8] = -200;
            }
            else
            {
                if(mines[8]/8 > this.density)
                {
                    map(mines[8], this.density, 1, 0, -0.6);
                }
                else
                {
                    map(mines[8], 0, this.density, 0.4, 0.05);
                }
            }
            Probability = Probability + mines[8];
            if (Probability > random(0, 1))
            {
                this.mineboard[current/this.board_width][current%this.board_width] = 1;
                times++;
            }
            Used.add(current);
        }
    }

    public int[] Detect_mines(int Situation[], int x, int y)
    {
        int[] location = new int[12]; // 0~7: no mine location, 8: mines quantities, 9: flag quantities, 10: flag wrong, 11:empty any side
        int[][] place = {{1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}};
        location[11] = 0;
        for (int i = 0 ; i < 8 ; i++)
        {
            location[i] = -1;
            if(Situation[i] != 0)
            {
                if(this.mineboard[x + place[Situation[i] - 1][0]][y + place[Situation[i] - 1][1]] == 1)
                {
                    location[8]++;                //mines quantities
                }
                else
                {
                    location[i] = (x + place[Situation[i] - 1][0])*this.board_width + y + place[Situation[i] - 1][1];      
                }
                if(this.covered[x + place[Situation[i] - 1][0]][y + place[Situation[i] - 1][1]] == 10)
                {
                    if(this.mineboard[x + place[Situation[i] - 1][0]][y + place[Situation[i] - 1][1]] == 0)
                    {
                        location[10] = 1;
                    }
                    location[9]++;                //flag quantities
                }
                if(this.covered[x + place[Situation[i] - 1][0]][y + place[Situation[i] - 1][1]] == 9)
                {
                    location[11] = 1;
                }
            }
        }
        return location;
    }

    public int[] Detect_array_avaliable(int current)   //2-axis array detecion
    {
        int detection[] = new int[8];
        if (current/this.board_width == 0)
        {
            if(current%this.board_width == 0)
            {
                for (int i = 0; i < 3 ; i++)
                {
                    detection[i] = i + 1;
                }
            }
            else if (current%this.board_width == this.board_width - 1)
            {
                detection[0] = 1;
                detection[1] = 7;
                detection[2] = 8;
            }
            else
            {
                detection[0] = 8;
                detection[1] = 1;
                detection[2] = 2;
                detection[3] = 3;
                detection[4] = 7;
            }
        }
        else if (current/this.board_width == this.board_height - 1)
        {
            if(current%this.board_width == 0)
            {
                for (int i = 0; i < 3 ; i++)
                {
                    detection[i] = i + 3;
                }
            }
            else if (current%this.board_width == this.board_width - 1)
            {
                for (int i = 0; i < 3 ; i++)
                {
                    detection[i] = i + 5;
                }
            }
            else
            {
                for (int i = 0; i < 4 ; i++)
                {
                    detection[i] = i + 4;
                }
                detection[4] = 3;
            }
        }
        else if (current%this.board_width == 0)
        {
            for (int i = 0; i < 5 ; i++)
            {
                detection[i] = i + 1;
            }
        }
        else if (current%this.board_width == this.board_width - 1)
        {
            for (int i = 0; i < 4 ; i++)
            {
                detection[i] = i + 5;
            }
            detection[4] = 1;
        }
        else
        {
            for (int i = 0; i < 8 ; i++)
            {
                detection[i] = i + 1;
            }
        }
        return detection;
    } 

    public void draw_covered()
    {   
        for (int y = 0 ; y < this.board_height; y++)
        {
            for (int x = 0 ; x < this.board_width; x++)
            {  
                int[] detect = Detect_mines(Detect_array_avaliable(y*this.board_width + x), y, x);
                stroke(0);
                int select = n_colors[this.covered[y][x]];
                if(select == 100)
                {
                    fill(#404040);
                    stroke(255);
                    rect(x*block_length + 2, 80 + y*block_width + 2, block_length - 4, block_width - 4);
                    fill(#C6C6C6);
                    beginShape(TRIANGLES);
                    vertex(x*block_length + 6, 80 + y*block_width + 4);
                    vertex(x*block_length + 6, 80 + y*block_width + 20);
                    vertex(x*block_length + 23, 80 + y*block_width + 13);
                    endShape();
                    line(x*block_length + 6, 80 + y*block_width + 4, x*block_length + 6, 80 + (y + 1)*block_width - 4);
                }
                else
                {
                    if(detect[9] > detect[8] && this.covered[y][x] != 0 && this.covered[y][x] != 9)
                    {
                        fill(n_colors[11]);
                    }
                    else
                    {
                        fill(select);
                    }
                    rect(x*block_length, 80 + y*block_width, block_length, block_width);
                    if(this.covered[y][x] > 0 && this.covered[y][x] < 9)
                    {
                        fill(0);
                        text(String.valueOf(this.covered[y][x]) , x*block_length + 7, 80 + y*block_width + 26);
                    }
                }
                // reveal mines
                // if(this.mineboard[y][x] == 1)
                // {
                //     fill(#FF2121);
                //     stroke(#FF2121);
                //     circle(x*block_length + block_length/2, 80 + y*block_width + block_width/2, block_length/2 + 2);
                //     strokeWeight(3);
                //     line(x*block_length + block_length/2, 80 + y*block_width + 2, x*block_length + block_length/2, 80 + y*block_width + 28);
                //     line(x*block_length + 4, 80 + y*block_width + 6, x*block_length + 26, 80 + y*block_width + 24);
                //     line(x*block_length + 26, 80 + y*block_width + 6, x*block_length + 4, 80 + y*block_width + 24);
                //     line(x*block_length + 2, 80 + y*block_width + block_width/2, (x+1)*block_length - 2, 80 + y*block_width + block_width/2);
                //     strokeWeight(1);
                // }
            }
        }
    }

    public int detect_location()  //current mouse location
    {
        int current = -1;
        stroke(#909090);
        for (int y = 0 ; y < this.board_height; y++)
        {
            for (int x = 0 ; x < this.board_width; x++)
            {  
                if (mouseX > x*block_length && mouseX < (x + 1)*block_length &&
                    mouseY > 80 + y*block_width && mouseY < 80 + (y + 1)*block_width)
                {
                    fill(#909090);
                    current = y * this.board_width + x;
                    rect(x*block_length + 2, 80 + y*block_width + 2, block_length - 4, block_width - 4);
                }
            }
        }
        return current;
    }
    
    public void create_safezone(int location)
    {
        ArrayList Used = new ArrayList();
        Used.add(location);
        int End = 0;
        int current = location;
        int next = 0;
        int[] B = Detect_mines(Detect_array_avaliable(current), current/this.board_width, current%this.board_width);
        if(B[9] > B[8] && this.covered[current/(this.board_width)][current%(this.board_width)] != 0)
        {
            End = 1;
        }
        else if (B[9] == B[8] && B[10] == 1 && B[11] == 1)
        {
            choose = 4;
            End = 1;
        }
        while(End == 0)
        {
            int[] Detect;
            current = (int)Used.get(next);
            Detect = Detect_mines(Detect_array_avaliable(current), current/this.board_width, current%this.board_width);
            if(Detect[8] > 0)
            {
                this.covered[current/(this.board_width)][current%(this.board_width)] = Detect[8];
            }
            if(Detect[9] == Detect[8] || Detect[8] == 0)
            {
                for (int i = 0 ; i < 8 ; i++)
                {
                    if(Detect[i] != -1 && Used.contains(Detect[i]) == false)
                    {   
                        int[] Detect_next = Detect_mines(Detect_array_avaliable(Detect[i]), Detect[i]/this.board_width, Detect[i]%this.board_width);
                        if(Detect_next[8] == 0)
                        {
                            Used.add(Detect[i]);
                        }
                        else
                        {
                            if(this.covered[Detect[i]/(this.board_width)][Detect[i]%(this.board_width)] != 10)
                            {
                                this.covered[Detect[i]/(this.board_width)][Detect[i]%(this.board_width)] = Detect_next[8];
                            }
                        }
                    }
                }
            }
            if(Detect[8] == 0)
            {
                Detect[8] = 9;
            }
            this.covered[(int)Used.get(next)/(this.board_width)][(int)Used.get(next)%(this.board_width)] = Detect[8];
            next++;
            if (next == Used.size())
            {
                End = 1;
            }
        }
    }

    public void Gaming()
    {
        if (initial_times == 0)
        {
            windowResize(this.board_width *block_length, this.board_height *block_width + 80);
            this.plant_mines(mode);
            this.remaining_mines = this.mines;
            initial_times = 1;
        }
        fill(255);
        textSize(36);
        text("Remaining mines : " + String.valueOf(this.remaining_mines), width/2 - 155, 50);
        this.draw_covered();
        int current = this.detect_location();
        if (mousePressed && release == 1 || Serial_data[0] != 0)
        {
            if(current != -1 && (mouseButton == LEFT || Serial_data[0] == 1) && this.covered[current/this.board_width][current%this.board_width] != 10)
            {
                if (this.mineboard[current/this.board_width][current%this.board_width] == 1)
                {
                    choose = 4;     //lose
                }
                else
                {
                    this.create_safezone(current);
                }
                release = 0;
            }
            if(current != -1 && (mouseButton == RIGHT || Serial_data[0] == 2))
            {
                if(this.covered[current/this.board_width][current%this.board_width] == 0)
                {
                    this.covered[current/this.board_width][current%this.board_width] = 10; // flag
                    this.remaining_mines--;
                }
                else if(this.covered[current/this.board_width][current%this.board_width] == 10)
                {
                    this.covered[current/this.board_width][current%this.board_width] = 0; // flag
                    this.remaining_mines++;
                }
                stroke(0);
                fill(0);
                rect(width/2 + 120, 20, 80, 40);
                release = 0;
            }
        }
        if(this.remaining_mines == 0)
        {
            for(short y = 0 ; y < this.board_height ; y++)
            {
                for(short x = 0 ; x < this.board_width ; x++)
                {
                    if(this.mineboard[y][x] == 1 && this.covered[y][x] == 10)
                    {
                        choose = 5;
                    }
                }
            }
        }
    }

    public void Clear()
    {
        this.remaining_mines = this.mines;
        for(short y = 0 ; y < this.board_height ; y++)
        {
            for(short x = 0 ; x < this.board_width ; x++)
            {
                this.mineboard[y][x] = 0;
                this.covered[y][x] = 0;
            }
        }
    }

    public void Showing_mines()
    {
        if(initial_times == 1)
        {
            this.draw_covered();
            for(short y = 0 ; y < this.board_height ; y++)
            {
                for(short x = 0 ; x < this.board_width ; x++)
                {
                    if(this.mineboard[y][x] == 1 && this.covered[y][x] !=10)
                    {
                        remain.add(y*this.board_width + x);
                    }
                }
            }
            initial_times = 2;
        }
        int explode = remain.get(int(random(0, remain.size() - 1)));
        fill(255);
        int x = explode%this.board_width;
        int y = explode/this.board_width;

        fill(#FF2121);
        stroke(#FF2121);
        circle(x*block_length + block_length/2, 80 + y*block_width + block_width/2, block_length/2 + 2);
        strokeWeight(3);
        line(x*block_length + block_length/2, 80 + y*block_width + 2, x*block_length + block_length/2, 80 + y*block_width + 28);
        line(x*block_length + 4, 80 + y*block_width + 6, x*block_length + 26, 80 + y*block_width + 24);
        line(x*block_length + 26, 80 + y*block_width + 6, x*block_length + 4, 80 + y*block_width + 24);
        line(x*block_length + 2, 80 + y*block_width + block_width/2, (x+1)*block_length - 2, 80 + y*block_width + block_width/2);
        strokeWeight(1);

        count++;
        delay(int(1000*logistic_function(map(count, 0, remain.size(), 0.5, 0.85))));
        remain.remove(Integer.valueOf(explode));
        if(remain.size() == 0)
        {
            delay(2000);
            choose = 6;
            count = 0;
        }
    }

    board(int x, int y, int m, float D)
    {
        board_width = x;
        board_height = y;
        mines = m;
        density = D;
        mineboard = new int[y][x];
        covered = new int[y][x];
    }
}

void setup()
{
    size(1080, 720);
    background(0);
    // USB_Port = new Serial(this, "COM4", 9600);
    // USB_Port.bufferUntil('\n');
}

// void serialEvent (Serial USB_Port) 
// {
//     float dataIn;
//     dataIn = float(USB_Port.readStringUntil('\n'));
//     Serial_data[0] = int(dataIn);
// }

void draw()
{   
    if (choose == 0)    //Clear
    {
        mode = 0;
        choose = 1;
        initial_times = 0;
        count = 0;
        remain.clear();
        background(0);
        stroke(0);
    }
    if (choose == 1)
    {
        textSize(72);
        fill(255);
        text("Minesweeper", width/2 - select_length/2 + 29, height/3); 
        fill(#BCBCBC);
        rect(width/2 - select_length/2, height/2 + 40, select_length, select_width);    //box start
        rect(width/2 - select_length/2, height/2 + 190, select_length, select_width);   //box exit game
        fill(#DEDEDE);
        textSize(50);
        text("Start game", width/2 - 115, height/2 + 105);
        text("Exit game", width/2 - 100, height/2 + 255); 
        if (select_pos(1))
        {
            fill(255);
            rect(width/2 - select_length/2, height/2 + 40, select_length, select_width);
            fill(#BCBCBC);
            rect(width/2 - select_length/2 + 3, height/2 + 40 + 3, select_length - 6, select_width - 6);
            fill(#EDEDED);
            textSize(50);
            text("Start game", width/2 - 115, height/2 + 105);
            if((mousePressed && release == 1) || Serial_data[0] == 1)
            {
                choose = 2;
                release = 0;
                delay(80);
            }
        }
        if (select_pos(2))
        {
            fill(255);
            rect(width/2 - select_length/2, height/2 + 190, select_length, select_width);
            fill(#BCBCBC);
            rect(width/2 - select_length/2 + 3, height/2 + 190 + 3, select_length - 6, select_width - 6);
            fill(#EDEDED);
            textSize(50);
            text("Exit game", width/2 - 100, height/2 + 255);
            if(mousePressed && release == 1 || Serial_data[0] == 1)
            {
                exit();
            }
        }
    }
    else if (choose == 2)
    {
        fill(#BCBCBC);
        rect(width/2 + select_length/2 + 20, height/2 + 40, select_length/2.5, select_width/2);
        rect(width/2 + select_length/2 + 20, height/2 + 120, select_length/2.5, select_width/2);
        rect(width/2 + select_length/2 + 20, height/2 + 200, select_length/2.5, select_width/2);
        fill(#DEDEDE);
        textSize(36);
        text("Choose Difficulty", width/2 + select_length/2 - 10, height/2);
        textSize(36);
        text("Easy", width/2 + select_length/2 + 80, height/2 + 75);
        text("Normal", width/2 + select_length/2 + 63, height/2 + 155); 
        text("Hard", width/2 + select_length/2 + 80, height/2 + 235);
        if (choose_pos(1))
        {
            fill(255);
            rect(width/2 + select_length/2 + 20, height/2 + 40, select_length/2.5, select_width/2);
            fill(#BCBCBC);
            rect(width/2 + select_length/2 + 23, height/2 + 43, select_length/2.5 - 6, select_width/2 - 6);
            fill(#EDEDED);
            textSize(36);
            text("Easy", width/2 + select_length/2 + 80, height/2 + 75);
            if(mousePressed && release == 1 || Serial_data[0] == 1)
            {
                mode = 1;
                choose = 3;
                release = 0;
                background(0);
            }
        }
        else if (choose_pos(2))
        {
            fill(255);
            rect(width/2 + select_length/2 + 20, height/2 + 120, select_length/2.5, select_width/2);
            fill(#BCBCBC);
            rect(width/2 + select_length/2 + 23, height/2 + 123, select_length/2.5 - 6, select_width/2 - 6);
            fill(#EDEDED);
            textSize(36);
            text("Normal", width/2 + select_length/2 + 63, height/2 + 155);
            if(mousePressed && release == 1 || Serial_data[0] == 1)
            {
                mode = 2;
                choose = 3;
                release = 0;
                background(0);
            }
        }
        else if (choose_pos(3))
        {
            fill(255);
            rect(width/2 + select_length/2 + 20, height/2 + 200, select_length/2.5, select_width/2);
            fill(#BCBCBC);
            rect(width/2 + select_length/2 + 23, height/2 + 203, select_length/2.5 - 6, select_width/2 - 6);
            fill(#EDEDED);
            textSize(36);
            text("Hard", width/2 + select_length/2 + 80, height/2 + 235);
            if(mousePressed && release == 1 || Serial_data[0] == 1)
            {
                mode = 3;
                choose = 3;
                release = 0;
                background(0);
            } 
        }
        else
        {
            if(mousePressed && release == 1 || Serial_data[0] == 1)
            {
                fill(0);
                rect(width/2 + select_length/2 + 20, height/2 - 30, select_length/2, height/2);
                rect(width/2 + select_length/2 - 20, height/2 - 30, select_length/2, 30);
                release = 0;
                choose = 1;
            } 
        }

    }
    else if (choose == 3)
    {
        switch (mode)
        {
            case 1:
            {
                Easy.Gaming();
                break;
            }
            case 2:
            {
                Normal.Gaming();
                break;
            }
            case 3:
            {
                Hard.Gaming();
                break;
            }
        }
    }
    else if (choose == 4)    //lose
    {
        switch (mode)
        {
            case 1:
            {
                Easy.Showing_mines();
                break;
            }
            case 2:
            {
                Normal.Showing_mines();
                break;
            }
            case 3:
            {
                Hard.Showing_mines();
                break;
            }
        }
    }
    else if (choose == 5)    //win
    {
        switch (mode)
        {
            case 1:
            {
                background(0);
                fill(count);
                stroke(count);
                textSize(48);
                text("Congratulations ! You win", 45, height/2 - 10);
                textSize(24);
                text("press left key anywhere to back to menu", width/4 - 50, height/2 + 40);
                if(mousePressed && mouseButton == LEFT && release == 1 || Serial_data[0] == 1)
                {
                    windowResize(1080, 720);
                    choose = 0;
                    Easy.Clear();
                }
                count++;
                delay(4);
                break;
            }
            case 2:
            {
                background(0);
                fill(count);
                stroke(count);
                textSize(64);
                text("Congratulations ! You win", width/6 - 15, height/2 - 10);
                textSize(32);
                text("press left key anywhere to back to menu", width/4 - 20, height/2 + 40);
                if(mousePressed && mouseButton == LEFT && release == 1 || Serial_data[0] == 1)
                {
                    windowResize(1080, 720);
                    choose = 0;
                    Normal.Clear();
                }
                count++;
                delay(4);
                break;
            }
            case 3:
            {
                background(0);
                fill(count);
                stroke(count);
                textSize(64);
                text("Congratulations ! You win", width/6 + 20, height/2 - 10);
                textSize(32);
                text("press left key anywhere to back to menu", width/4 + 5, height/2 + 50);
                if(mousePressed && mouseButton == LEFT && release == 1 || Serial_data[0] == 1)
                {
                    windowResize(1080, 720);
                    choose = 0;
                    Hard.Clear();
                }
                count++;
                delay(4);
                break;
            }
        }
    }
    else if (choose == 6)    //lose window
    {
        switch (mode)
        {
            case 1:
            {
                background(0);
                fill(count);
                stroke(count);
                textSize(48);
                text("You lose...", width/3, height/2 - 20);
                textSize(24);
                text("press left key anywhere to back to menu", width/4 - 50, height/2 + 30);
                if(mousePressed && mouseButton == LEFT && release == 1 || Serial_data[0] == 1)
                {
                    windowResize(1080, 720);
                    choose = 0;
                    Easy.Clear();
                }
                count++;
                delay(4);
                break;
            }
            case 2:
            {
                background(0);
                fill(count);
                stroke(count);
                textSize(64);
                text("You lose...", width/3 + 40, height/2 - 10);
                textSize(32);
                text("press left key anywhere to back to menu", width/4 - 20, height/2 + 40);
                if(mousePressed && mouseButton == LEFT && release == 1 || Serial_data[0] == 1)
                {
                    windowResize(1080, 720);
                    choose = 0;
                    Normal.Clear();
                }
                count++;
                delay(4);
                break;
            }
            case 3:
            {
                background(0);
                fill(count);
                stroke(count);
                textSize(72);
                text("You lose...", width/3 + 35, height/2 - 15);
                textSize(38);
                text("press left key anywhere to back to menu", width/4 - 40, height/2 + 55);
                if(mousePressed && mouseButton == LEFT && release == 1 || Serial_data[0] == 1)
                {
                    windowResize(1080, 720);
                    choose = 0;
                    Hard.Clear();
                }
                count++;
                delay(4);
                break;
            }
        }
    }
}

boolean select_pos(int select)
{
    if (select == 1)
    {
        if (mouseX > width/2 - select_length/2 && mouseX < width/2 + select_length/2 &&
        mouseY > height/2 + 40 && mouseY < height/2 + 40 + select_width)
        {
            return true;
        }
        else 
        {
            return false;
        }
    }
    else
    {
        if (mouseX > width/2 - select_length/2 && mouseX < width/2 + select_length/2 &&
        mouseY > height/2 + 190 && mouseY < height/2 + 190 + select_width)
        {
            return true;
        }
        else 
        {
            return false;
        }
    }
}

boolean choose_pos(int select)
{
    if (select == 1)
    {
        if (mouseX > width/2 + select_length/2 + 20 && mouseX < width/2 + 9 * select_length/10+ 20 &&
        mouseY > height/2 + 40 && mouseY < height/2 + 40 + select_width/2)
        {
            return true;
        }
        else 
        {
            return false;
        }
    }
    else if (select == 2)
    {
        if (mouseX > width/2 + select_length/2 + 20 && mouseX < width/2 + 9 * select_length/10+ 20 &&
        mouseY > height/2 + 120 && mouseY < height/2 + 120 + select_width/2)
        {
            return true;
        }
        else 
        {
            return false;
        }
    }
    else
    {
        if (mouseX > width/2 + select_length/2 + 20 && mouseX < width/2 + 9 * select_length/10+ 20 &&
        mouseY > height/2 + 200 && mouseY < height/2 + 200 + select_width/2)
        {
            return true;
        }
        else 
        {
            return false;
        }
    }
}

float logistic_function(float x)
{
    return (1/(1 + exp(-x + 5)));
}

void mouseReleased() 
{
    release = 1;
}
